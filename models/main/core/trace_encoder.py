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
        # PyTorch 2.x nested-tensor fast path can produce NaNs in training when using
        # src_key_padding_mask (see nested tensor warning in transformer.py). Disable it.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        
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
        
        # src_key_padding_mask: True = ignore position. If a row has no valid tokens,
        # the mask is all-True and PyTorch TransformerEncoder returns NaNs for that row
        # (forward + backward). Add a sentinel "valid" slot at index 0 for those rows only
        # so the encoder never sees an all-masked row; pooling still uses the original mask.
        if mask is None:
            padding_mask = None
        else:
            valid_count = mask.sum(dim=1)
            empty_seq = valid_count == 0
            if empty_seq.any():
                mask_for_transformer = mask.clone()
                mask_for_transformer[empty_seq, 0] = True
                padding_mask = ~mask_for_transformer
            else:
                padding_mask = ~mask

        transformed = self.transformer(x, src_key_padding_mask=padding_mask)
        # (batch_size, seq_len, hidden_dim)
        if mask is not None:
            transformed = torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

        # Aggregate: use CLS-like mechanism (mean pooling over valid tokens)
        if mask is not None:
            # Mask-aware pooling
            mask_expanded = mask.unsqueeze(-1).expand(transformed.size())  # (batch_size, seq_len, hidden_dim)
            sum_emb = (transformed * mask_expanded).sum(dim=1)  # (batch_size, hidden_dim)
            sum_mask = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
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
        batch_size = x.size(0)
        hidden_dim = x.size(2)

        # Pack sequence for LSTM (length 0 is invalid for pack_padded_sequence)
        if mask is not None:
            seq_lengths = mask.sum(dim=1).long()
            output = torch.zeros(batch_size, hidden_dim, device=x.device, dtype=x.dtype)
            valid = seq_lengths > 0
            if valid.any():
                x_v = x[valid]
                lens_v = seq_lengths[valid].cpu()
                x_packed = nn.utils.rnn.pack_padded_sequence(
                    x_v, lens_v, batch_first=True, enforce_sorted=False
                )
                _, (h_n, c_n) = self.lstm(x_packed)
                output[valid] = h_n[-1]
            # rows with empty trace stay zeros
        else:
            _, (h_n, c_n) = self.lstm(x)
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
                output = masked_x.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-9)
            elif self.pool_type == "max":
                masked_x = x.clone()
                masked_x[~mask_expanded] = float('-inf')
                output = torch.max(masked_x, dim=1)[0]
                # All-padding rows: max is -inf → would break downstream layers
                empty_rows = mask.sum(dim=1) == 0
                if empty_rows.any():
                    output[empty_rows] = 0
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

    Eight semantic dimensions per call event:
    - Call type          (discrete → Embedding)
    - Callee address     (hash % N → Embedding)
    - Function selector  (hash % N → Embedding)
    - Call depth         (integer → Embedding)
    - Execution status   (0: empty output / 1: has output → Embedding)
    - Input size         (log-normalised scalar → Linear)
    - Output size        (log-normalised scalar → Linear)
    - Gas usage          (log-normalised scalar → Linear)
    """

    def __init__(
        self,
        call_type_vocab_size: int = 10,
        contract_vocab_size: int = 50000,
        func_selector_vocab_size: int = 100000,
        depth_max: int = 50,
        embedding_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Discrete → Embedding
        self.call_type_emb = nn.Embedding(call_type_vocab_size, embedding_dim, padding_idx=0)
        self.contract_emb = nn.Embedding(contract_vocab_size, embedding_dim, padding_idx=0)
        self.func_selector_emb = nn.Embedding(func_selector_vocab_size, embedding_dim, padding_idx=0)
        # depth / status: must reserve index 0 for padding only. Data stores raw depth in [0, depth_max-1]
        # and status in {0,1}; we shift by +1 in forward when trace_mask marks valid positions.
        self.depth_max = depth_max
        self.depth_emb = nn.Embedding(depth_max + 1, embedding_dim, padding_idx=0)
        # 0 = pad, 1 = status 0 (empty output), 2 = status 1
        self.status_emb = nn.Embedding(3, embedding_dim, padding_idx=0)

        # Continuous scalar → Linear projection
        self.input_size_proj = nn.Sequential(
            nn.Linear(1, embedding_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.output_size_proj = nn.Sequential(
            nn.Linear(1, embedding_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.gas_proj = nn.Sequential(
            nn.Linear(1, embedding_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # Combined output dimension: 8 semantic components × embedding_dim
        self.output_dim = 8 * embedding_dim

    def forward(
        self,
        call_type_ids: torch.Tensor,       # (batch, seq_len)
        contract_ids: torch.Tensor,        # (batch, seq_len)
        func_selector_ids: torch.Tensor,   # (batch, seq_len)
        depths: torch.Tensor,              # (batch, seq_len)
        status_ids: torch.Tensor,          # (batch, seq_len)  0 or 1
        input_sizes: torch.Tensor,         # (batch, seq_len)  log-normalised
        output_sizes: torch.Tensor,        # (batch, seq_len)  log-normalised
        gas_vals: torch.Tensor,             # (batch, seq_len)  log-normalised
        trace_mask: Optional[torch.Tensor] = None,  # (batch, seq_len) True = valid token
    ) -> torch.Tensor:
        """
        Returns:
            (batch, seq_len, 8 * embedding_dim) - combined event embeddings
        """
        type_emb     = self.call_type_emb(call_type_ids)
        contract_emb = self.contract_emb(contract_ids)
        func_emb     = self.func_selector_emb(func_selector_ids)

        depth_clamped = depths.long().clamp(0, self.depth_max - 1)
        depth_shifted = depth_clamped + 1  # 1..depth_max for real depths 0..depth_max-1
        if trace_mask is not None:
            depth_ids = torch.where(trace_mask, depth_shifted, torch.zeros_like(depth_shifted))
        else:
            depth_ids = depth_shifted

        status_clamped = status_ids.long().clamp(0, 1)
        status_shifted = status_clamped + 1
        if trace_mask is not None:
            status_ids_emb = torch.where(trace_mask, status_shifted, torch.zeros_like(status_shifted))
        else:
            status_ids_emb = status_shifted

        depth_emb = self.depth_emb(depth_ids)
        status_out = self.status_emb(status_ids_emb)

        input_emb  = self.input_size_proj(input_sizes.unsqueeze(-1))
        output_emb = self.output_size_proj(output_sizes.unsqueeze(-1))
        gas_emb    = self.gas_proj(gas_vals.unsqueeze(-1))

        combined = torch.cat(
            [type_emb, contract_emb, func_emb, depth_emb,
             status_out, input_emb, output_emb, gas_emb],
            dim=-1
        )
        return combined
