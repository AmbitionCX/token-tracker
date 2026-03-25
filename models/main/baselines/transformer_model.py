"""
Baseline 4: Transformer (Trace Only)

Uses ONLY internal call trace features (no external features, no graph).
The call event sequence is embedded, then processed by a Transformer encoder.
A [CLS] token is prepended; its final representation is used for classification.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """Transformer encoder on call trace sequences for binary classification."""

    def __init__(
        self,
        input_dim: int = 128,       # CallEventEmbedding output dim
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project input to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Positional encoding (max_len + 1 for CLS)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_seq_len + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        trace_embeddings: torch.Tensor,                # (B, L, input_dim)
        trace_mask: Optional[torch.Tensor] = None,     # (B, L)  bool, True=valid
    ) -> torch.Tensor:
        """
        Args:
            trace_embeddings: per-token embeddings from CallEventEmbedding
            trace_mask: True for valid positions

        Returns:
            (B, num_classes) logits
        """
        B, L, _ = trace_embeddings.shape

        x = self.input_proj(trace_embeddings)              # (B, L, hidden_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)      # (B, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)              # (B, 1+L, hidden_dim)

        x = self.pos_enc(x)

        # Build key_padding_mask: True = ignored position (PyTorch convention)
        if trace_mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, trace_mask], dim=1)  # (B, 1+L)
            key_padding_mask = ~full_mask  # True = pad → ignore
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Take [CLS] representation
        cls_repr = x[:, 0, :]                              # (B, hidden_dim)

        return self.classifier(cls_repr)
