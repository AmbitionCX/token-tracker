"""
Baseline 3: LSTM (Trace Only)

Uses ONLY internal call trace features (no external features, no graph).
The call event sequence is embedded via CallEventEmbedding, then fed into
a bidirectional LSTM.  The final hidden state is used for classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMBaseline(nn.Module):
    """Bidirectional LSTM on call trace sequences for binary classification."""

    def __init__(
        self,
        input_dim: int = 128,       # CallEventEmbedding output dim
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # bidirectional → 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        trace_embeddings: torch.Tensor,                # (B, L, input_dim)
        trace_mask: Optional[torch.Tensor] = None,     # (B, L)  bool
    ) -> torch.Tensor:
        """
        Args:
            trace_embeddings: per-token embeddings from CallEventEmbedding
            trace_mask: True for valid positions

        Returns:
            (B, num_classes) logits
        """
        x = self.input_proj(trace_embeddings)          # (B, L, hidden_dim)

        if trace_mask is not None:
            # Pack padded sequence for efficient LSTM
            lengths = trace_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # h_n: (num_layers * 2, B, hidden_dim) → take last layer fwd + bwd
        h_fwd = h_n[-2]   # last layer forward
        h_bwd = h_n[-1]   # last layer backward
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)     # (B, hidden_dim*2)

        return self.classifier(h_cat)
