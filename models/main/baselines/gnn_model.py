"""
GNN baseline — external transaction features only (no trace encoding).

Uses :class:`NoTraceSequenceGNN` from ``seq_gnn_model``: GNN message passing on
external + zero-padded trace dimension (trace branch disabled).
"""

from models.seq_gnn_model import NoTraceSequenceGNN


class GNNExternalOnly(NoTraceSequenceGNN):
    """GNN with external features only (trace sequence encoder off)."""

    pass


__all__ = ["GNNExternalOnly"]
