"""
Baseline models for comparison.
Includes:
- Traditional ML: LogisticRegression, RandomForest, XGBoost
- Graph-only: GCN, GraphSAGE, GAT (no sequence encoding)
- Sequence-only: RNN, Transformer (no GNN)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from .xgboost_model import XGBoostBaseline
from .mlp_model import MLPBaseline
from .lstm_model import LSTMBaseline
from .transformer_model import TransformerBaseline


class TraditionalBaselines:
    """Traditional ML models on hand-crafted features (external only)."""
    
    @staticmethod
    def logistic_regression(**kwargs):
        """Logistic Regression baseline."""
        return LogisticRegression(max_iter=1000, **kwargs)
    
    @staticmethod
    def random_forest(**kwargs):
        """Random Forest baseline."""
        return RandomForestClassifier(n_estimators=100, **kwargs)
    
    @staticmethod
    def xgboost(**kwargs):
        """XGBoost baseline."""
        if 'objective' not in kwargs:
            kwargs['objective'] = 'binary:logistic'
        return xgb.XGBClassifier(**kwargs)


__all__ = [
    'TraditionalBaselines',
    'XGBoostBaseline',
    'MLPBaseline',
    'LSTMBaseline',
    'TransformerBaseline',
]