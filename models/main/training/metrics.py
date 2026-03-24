"""
Metrics computation for evaluation.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score, roc_curve, auc
)
from typing import Dict, Tuple


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    return_curves: bool = False
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted probabilities (0-1) or class indices
        labels: Ground truth binary labels (0/1)
        threshold: Classification threshold (for soft predictions)
        return_curves: Whether to return ROC curve (requires probabilities)
    
    Returns:
        Dictionary of metrics
    """
    # Convert soft predictions to binary
    if predictions.max() <= 1.0 and predictions.min() >= 0.0 and predictions.dtype != np.int32:
        binary_preds = (predictions >= threshold).astype(int)
        soft_preds = predictions
    else:
        binary_preds = predictions.astype(int)
        soft_preds = None
    
    metrics = {
        'accuracy': accuracy_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds, zero_division=0),
        'recall': recall_score(labels, binary_preds, zero_division=0),
        'f1': f1_score(labels, binary_preds, zero_division=0),
    }
    
    # AUC (requires soft predictions)
    if soft_preds is not None:
        try:
            metrics['auc'] = roc_auc_score(labels, soft_preds)
        except:
            metrics['auc'] = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return metrics


__all__ = [
    'compute_metrics',
]
