"""
Trace Encoder: Internal call trace sequence encoding

Converts linearized call trace sequences into fixed-dimension representations
using Transformer, LSTM, or simple pooling aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTraceEncoder(nn.Module):
    """
    Trace encoder using Transformer architecture.
    
    Captures long-range dependencies between call events via self-attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=256, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (for aggregation)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - call event embeddings
            mask: (batch_size, seq_len) - boolean mask (True for valid positions)
        
        Returns:
            (batch_size, hidden_dim) - aggregated trace representation
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask for padding
        # PyTorch uses True for positions to MASK OUT (ignore)
        if mask is None:
            attn_mask = None
        else:
            # mask: True for valid, False for padding
            # attn_mask: True for masking out (inverse)
            attn_mask = ~mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
        
        # Apply transformer
        transformed = self.transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        # (batch_size, seq_len, hidden_dim)
        
        # Aggregate: use CLS-like mechanism (mean pooling over valid tokens)
        if mask is not None:
            # Mask-aware pooling
            mask_expanded = mask.unsqueeze(-1).expand(transformed.size())  # (batch_size, seq_len, hidden_dim)
            sum_emb = (transformed * mask_expanded).sum(dim=1)  # (batch_size, hidden_dim)
            sum_mask = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # (batch_size, 1, 1)
            output = sum_emb / (sum_mask + 1e-9)
        else:
            # Simple mean pooling over sequence
            output = transformed.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Final projection
        output = self.output_proj(output)  # (batch_size, hidden_dim)
        
        return output


class LSTMTraceEncoder(nn.Module):
    """
    Trace encoder using LSTM architecture.
    
    Alternative to Transformer for sequential modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - call event embeddings
            mask: (batch_size, seq_len) - boolean mask (True for valid positions)
        
        Returns:
            (batch_size, hidden_dim) - aggregated trace representation
        """
        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Pack sequence for LSTM
        if mask is not None:
            seq_lengths = mask.sum(dim=1).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        # h_n: (num_layers, batch_size, hidden_dim)
        output = h_n[-1]  # (batch_size, hidden_dim)
        
        # Final projection
        output = self.output_proj(output)  # (batch_size, hidden_dim)
        
        return output


class PoolingTraceEncoder(nn.Module):
    """
    Simple pooling-based aggregation for trace sequences.
    
    Baseline: no sequential modeling, just feature aggregation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pool_type: str = "mean"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        
        # Input processing
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - call event embeddings
            mask: (batch_size, seq_len) - boolean mask (True for valid positions)
        
        Returns:
            (batch_size, hidden_dim) - aggregated trace representation
        """
        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        x = self.norm(x)
        x = self.dropout(x)
        
        # Pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(x.size())
            masked_x = x * mask_expanded
            
            if self.pool_type == "mean":
                output = masked_x.sum(dim=1) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-9)
            elif self.pool_type == "max":
                masked_x[~mask_expanded] = float('-inf')
                output = torch.max(masked_x, dim=1)[0]
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")
        else:
            if self.pool_type == "mean":
                output = x.mean(dim=1)
            elif self.pool_type == "max":
                output = torch.max(x, dim=1)[0]
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Final projection
        output = self.output_proj(output)
        
        return output


class TraceEncoder(nn.Module):
    """
    Unified Trace Encoder interface.
    
    Factory pattern to support multiple encoding strategies:
    - Transformer: self-attention over call sequence
    - LSTM: recurrent sequential encoding
    - Pooling: simple aggregation baseline
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        encoder_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        
        if encoder_type == "transformer":
            self.encoder = TransformerTraceEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation
            )
        elif encoder_type == "lstm":
            self.encoder = LSTMTraceEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif encoder_type == "pooling":
            self.encoder = PoolingTraceEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                pool_type="mean"
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - call event embeddings
            mask: (batch_size, seq_len) - boolean mask (True for valid positions)
        
        Returns:
            (batch_size, hidden_dim) - trace representation
        """
        return self.encoder(x, mask)


# ============================================================================
# Call Event Embedding Layer (preprocessing before TraceEncoder)
# ============================================================================

class CallEventEmbedding(nn.Module):
    """
    Convert raw call trace information into dense embeddings.
    
    Embeds multiple semantic dimensions of a call event:
    - Call type (CALL, DELEGATECALL, STATICCALL, CREATE)
    - Called contract address
    - Function selector (4-byte)
    - Call depth
    - Execution properties (revert status, gas used, I/O size)
    """
    
    def __init__(
        self,
        call_type_vocab_size: int = 10,      # Types: CALL, DELEGATECALL, STATICCALL, CREATE2, etc.
        contract_vocab_size: int = 50000,    # Number of unique contracts
        func_selector_vocab_size: int = 100000,  # Number of unique functions
        depth_max: int = 50,                  # Max call depth
        embedding_dim: int = 32,              # Dimension per semantic component
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Semantic dimension embeddings
        self.call_type_emb = nn.Embedding(call_type_vocab_size, embedding_dim, padding_idx=0)
        self.contract_emb = nn.Embedding(contract_vocab_size, embedding_dim, padding_idx=0)
        self.func_selector_emb = nn.Embedding(func_selector_vocab_size, embedding_dim, padding_idx=0)
        self.depth_emb = nn.Embedding(depth_max, embedding_dim)
        
        # Execution properties: [revert_flag, log_input_size, log_output_size, log_gas_used]
        self.exec_proj = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined output dimension: 5 embeddings × embedding_dim
        self.output_dim = 5 * embedding_dim
    
    def forward(
        self,
        call_type_ids: torch.Tensor,        # (batch_size, seq_len)
        contract_ids: torch.Tensor,         # (batch_size, seq_len)
        func_selector_ids: torch.Tensor,    # (batch_size, seq_len)
        depths: torch.Tensor,               # (batch_size, seq_len)
        exec_properties: torch.Tensor       # (batch_size, seq_len, 4)
    ) -> torch.Tensor:
        """
        Args:
            call_type_ids: Call type indices
            contract_ids: Contract address indices
            func_selector_ids: Function selector indices
            depths: Call depth (0-49)
            exec_properties: [revert_flag, log_input_size, log_output_size, log_gas_used]
        
        Returns:
            (batch_size, seq_len, 5*embedding_dim) - combined event embeddings
        """
        # Embed each semantic dimension
        type_emb = self.call_type_emb(call_type_ids)  # (batch, seq, embed)
        contract_emb = self.contract_emb(contract_ids)
        func_emb = self.func_selector_emb(func_selector_ids)
        depth_emb = self.depth_emb(depths.long())
        
        # Embed execution properties
        exec_emb = self.exec_proj(exec_properties)  # (batch, seq, embed)
        
        # Concatenate all embeddings
        combined = torch.cat([type_emb, contract_emb, func_emb, depth_emb, exec_emb], dim=-1)
        # (batch_size, seq_len, 5*embedding_dim)
        
        return combined
