"""
Ablation experiment configurations.

4 main ablation experiments:
- Exp A: Remove Trace Features (GNN importance)
- Exp B: Remove GNN module (Graph importance)
- Exp C: Remove Attention (Attention mechanism importance)
- Exp D: Compare TraceEncoder types (Transformer vs LSTM vs Pooling)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class AblationConfig:
    """Configuration for a single ablation / baseline experiment."""
    
    name: str
    description: str
    use_trace: bool = True
    use_gnn: bool = True
    use_attention: bool = True
    use_external: bool = True   # False → "Trace only" models (no external features)
    trace_encoder_type: str = "transformer"
    gnn_type: str = "gat"
    # 'sklearn' → handled separately (xgboost_baseline.py)
    model_backend: str = "pytorch"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'use_trace': self.use_trace,
            'use_gnn': self.use_gnn,
            'use_attention': self.use_attention,
            'use_external': self.use_external,
            'trace_encoder_type': self.trace_encoder_type,
            'gnn_type': self.gnn_type,
            'model_backend': self.model_backend,
        }


# Define ablation experiments
ABLATION_CONFIGS = {
    'complete_model': AblationConfig(
        name="Complete Model",
        description="Full SequenceGNN with Trace + GNN + Attention",
        use_trace=True,
        use_gnn=True,
        use_attention=True,
        trace_encoder_type="transformer"
    ),
    
    'no_trace': AblationConfig(
        name="No Trace Features",
        description="GNN + Attention but no internal call trace features",
        use_trace=False,
        use_gnn=True,
        use_attention=True
    ),
    
    'no_gnn': AblationConfig(
        name="No GNN Module",
        description="Trace encoding direct to classification, no graph",
        use_trace=True,
        use_gnn=False,
        use_attention=True
    ),
    
    'no_attention': AblationConfig(
        name="No Attention Aggregation",
        description="Trace + GNN but simple concatenation aggregation",
        use_trace=True,
        use_gnn=True,
        use_attention=False
    ),
    
    'lstm_encoder': AblationConfig(
        name="LSTM TraceEncoder",
        description="Replace Transformer with LSTM for call trace encoding",
        use_trace=True,
        use_gnn=True,
        use_attention=True,
        trace_encoder_type="lstm"
    ),
    
    'pooling_encoder': AblationConfig(
        name="Pooling TraceEncoder",
        description="Replace Transformer with simple pooling for call trace",
        use_trace=True,
        use_gnn=True,
        use_attention=True,
        trace_encoder_type="pooling"
    ),
    
    'gcn_gnn': AblationConfig(
        name="GCN instead of GAT",
        description="Use Graph Convolutional Network instead of GAT",
        use_trace=True,
        use_gnn=True,
        use_attention=True,
        trace_encoder_type="transformer",
        gnn_type="gcn"
    ),
}


# Baseline models (for comparison with ablations)
BASELINE_CONFIGS = {
    'logistic_regression': {
        'name': 'Logistic Regression',
        'type': 'sklearn',
        'description': 'Traditional ML on external features only',
        'features': 'external_only'
    },
    
    'random_forest': {
        'name': 'Random Forest',
        'type': 'sklearn',
        'description': 'Ensemble tree model on external features',
        'features': 'external_only'
    },
    
    'xgboost': {
        'name': 'XGBoost',
        'type': 'sklearn',
        'description': 'Gradient boosting on external features',
        'features': 'external_only'
    },
}


def get_ablation_config(experiment_id: str) -> AblationConfig:
    """Get ablation configuration by ID."""
    if experiment_id not in ABLATION_CONFIGS:
        raise KeyError(f"Unknown experiment: {experiment_id}")
    return ABLATION_CONFIGS[experiment_id]


def list_ablation_experiments() -> List[str]:
    """List all ablation experiment IDs."""
    return list(ABLATION_CONFIGS.keys())


__all__ = [
    'AblationConfig',
    'ABLATION_CONFIGS',
    'BASELINE_CONFIGS',
    'get_ablation_config',
    'list_ablation_experiments',
]
