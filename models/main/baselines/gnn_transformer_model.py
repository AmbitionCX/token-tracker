"""
GNN + Transformer trace baseline.

Full sequence encoder (Transformer) + GNN — same architecture family as the
proposed main model, exposed as a named baseline for Table~1-style comparisons.
"""

from models.seq_gnn_model import SequenceGNNModel


class GNNTransformer(SequenceGNNModel):
    """GNN with Transformer trace encoder (``trace_encoder_type`` from config)."""

    pass


__all__ = ["GNNTransformer"]
