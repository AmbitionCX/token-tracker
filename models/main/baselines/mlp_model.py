"""
Baseline 2: MLP (External + Mean Trace)

Uses external features (4 dims) concatenated with mean-pooled trace
embeddings. The trace embeddings come from the project's CallEventEmbedding
layer, averaged over the sequence → 128 dims.

Total input = 4 (external) + 128 (mean trace) = 132 dims.
A simple 3-layer MLP classifier.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class MLPBaseline(nn.Module):
    """MLP classifier on external + mean-pooled trace features."""

    def __init__(
        self,
        input_dim: int = 132,       # 4 external + 128 trace
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        external_features: torch.Tensor,       # (B, 4)
        trace_features: torch.Tensor,           # (B, 128)
    ) -> torch.Tensor:
        """
        Args:
            external_features: (B, 4) normalised external features
            trace_features: (B, 128) mean-pooled trace embeddings

        Returns:
            (B, num_classes) logits
        """
        x = torch.cat([external_features, trace_features], dim=-1)
        return self.net(x)
