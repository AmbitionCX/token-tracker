"""
Core components for edge feature extraction and graph construction.
"""

from .trace_encoder import (
    TraceEncoder,
    TransformerTraceEncoder,
    LSTMTraceEncoder,
    PoolingTraceEncoder,
    CallEventEmbedding,
    PositionalEncoding
)

from .edge_feature_extractor import (
    EdgeFeatureExtractor,
    ExternalTransactionFeatures,
    InternalCallTraceFeatures,
    EdgeFeatureMask
)

from .temporal_graph_builder import (
    TemporalGraphBuilder,
    TemporalWindow,
    MultiWindowGraphDataset
)

from .attention_aggregator import (
    EdgeAttentionAggregator,
    EdgeRepresentationAggregator,
    NeighborhoodContextExtractor,
    build_neighborhood_graph
)

__all__ = [
    # Trace Encoding
    'TraceEncoder',
    'TransformerTraceEncoder',
    'LSTMTraceEncoder',
    'PoolingTraceEncoder',
    'CallEventEmbedding',
    'PositionalEncoding',
    
    # Edge Features
    'EdgeFeatureExtractor',
    'ExternalTransactionFeatures',
    'InternalCallTraceFeatures',
    'EdgeFeatureMask',
    
    # Temporal Graph
    'TemporalGraphBuilder',
    'TemporalWindow',
    'MultiWindowGraphDataset',
    
    # Attention & Aggregation
    'EdgeAttentionAggregator',
    'EdgeRepresentationAggregator',
    'NeighborhoodContextExtractor',
    'build_neighborhood_graph',
]
