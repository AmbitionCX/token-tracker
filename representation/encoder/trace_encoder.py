# representation/encoder/trace_encoder.py

import torch
import torch.nn as nn


class TraceEncoder(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # [B, L, d]
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, X, attn_mask=None):
        """
        X: [B, L, d]
        attn_mask: [B, L] (optional, padding mask)
        """
        H = self.encoder(
            X,
            src_key_padding_mask=attn_mask
        )
        return H
