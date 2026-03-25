"""
Baseline 1: XGBoost (External Features Only)

Uses only 4 external transaction features:
  - value (log-normalised ETH amount)
  - gas_used (log-normalised)
  - calldata_length (log-normalised)
  - is_revert (binary)

This is a non-neural, non-graph, non-trace baseline.
"""

import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from typing import Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from training.metrics import topk_precision_recall_metrics


class XGBoostBaseline:
    """XGBoost classifier on external transaction features only."""

    def __init__(self, pos_neg_ratio: float = 1.0, random_state: int = 42):
        """
        Args:
            pos_neg_ratio: ratio of negative / positive samples.
                           Used as scale_pos_weight to handle class imbalance.
            random_state: Random seed.
        """
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_neg_ratio,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        self.model.fit(X_train, y_train, **fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        out: Dict[str, float] = {
            "Accuracy": accuracy_score(y, y_pred),
            "PR_AUC": average_precision_score(y, y_prob) if y.sum() > 0 else 0.0,
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1_Score": f1_score(y, y_pred, zero_division=0),
            "AUC_ROC": roc_auc_score(y, y_prob) if y.sum() > 0 else 0.0,
        }
        tk = topk_precision_recall_metrics(y, y_prob, ks=(10, 50, 100))
        for k in (10, 50, 100):
            out[f"Precision@{k}"] = tk[f"precision_at_{k}"]
            out[f"Recall@{k}"] = tk[f"recall_at_{k}"]
        return out
