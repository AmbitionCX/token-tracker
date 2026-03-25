"""
GNN + mean-pooled trace baseline.

Uses :class:`PoolingSequenceGNN` (pooling trace encoder, typically mean over valid
positions) combined with the same GNN + edge classifier stack as the main model.
"""

from models.seq_gnn_model import PoolingSequenceGNN


class GNNMeanTrace(PoolingSequenceGNN):
    """GNN with mean-pooled (pooling encoder) trace representation."""

    pass


__all__ = ["GNNMeanTrace"]
