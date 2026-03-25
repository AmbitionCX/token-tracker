"""
Metrics computation for evaluation.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
)
from typing import Dict, Any, Optional, List, Tuple


def topk_precision_recall_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: Tuple[int, ...] = (10, 50, 100),
) -> Dict[str, float]:
    """
    Ranking-based metrics: sort by ``y_score`` descending, take top ``k`` samples.

    - **Precision@K**: (# positives in top-K) / K  (uses K_eff = min(K, n))
    - **Recall@K**: (# positives in top-K) / (total positives in dataset)
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n = y_true.size
    out: Dict[str, float] = {}
    if n == 0:
        for k in ks:
            out[f"precision_at_{k}"] = 0.0
            out[f"recall_at_{k}"] = 0.0
        return out

    order = np.argsort(-y_score)
    total_pos = int(y_true.sum())

    for k in ks:
        k_eff = min(int(k), n)
        top_idx = order[:k_eff]
        tp_k = int(y_true[top_idx].sum())
        out[f"precision_at_{k}"] = float(tp_k / k_eff) if k_eff > 0 else 0.0
        out[f"recall_at_{k}"] = float(tp_k / total_pos) if total_pos > 0 else 0.0
    return out


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    return_curves: bool = False,
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
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "macro_f1": f1_score(labels, binary_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, binary_preds, average="weighted", zero_division=0),
    }

    # AUC (requires soft predictions)
    if soft_preds is not None:
        try:
            metrics["auc"] = roc_auc_score(labels, soft_preds)
        except Exception:
            metrics["auc"] = 0.0
        try:
            metrics["pr_auc"] = average_precision_score(labels, soft_preds)
        except Exception:
            metrics["pr_auc"] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp

    # Additional metrics
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return metrics


def best_f1_threshold_from_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Choose threshold by maximizing F1 on the precision–recall curve (not fixed 0.5).

    ``precision_recall_curve`` returns ``len(thresholds) == len(precision) - 1``;
    F1 is evaluated on the first ``len(thresholds)`` points (``f1[:-1]``).
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0:
        return {
            "best_threshold": 0.5,
            "best_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
        }
    if len(np.unique(y_true)) < 2:
        # Single class: no meaningful PR curve
        thr = 0.5
        pred = (y_score >= thr).astype(int)
        return {
            "best_threshold": float(thr),
            "best_f1": float(
                f1_score(y_true, pred, zero_division=0)
            ),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, pred)),
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + eps)
    # Align with sklearn: one fewer threshold than precision points
    f1_valid = f1[:-1]
    if len(f1_valid) == 0 or len(thresholds) == 0:
        thr = 0.5
    else:
        best_idx = int(np.argmax(f1_valid))
        best_idx = min(best_idx, len(thresholds) - 1)
        thr = float(thresholds[best_idx])

    pred = (y_score >= thr).astype(int)
    return {
        "best_threshold": thr,
        "best_f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def compute_binary_test_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_fixed: float = 0.5,
    eps: float = 1e-8,
    topk_ks: Tuple[int, ...] = (10, 50, 100),
) -> Dict[str, float]:
    """
    Full test-set metrics: threshold-free AUCs, fixed-threshold, and PR-curve best-F1 threshold.

    Use ``y_score`` = P(class=1) (e.g. softmax[:, 1]).
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    out: Dict[str, float] = {}

    # Threshold-free
    if len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    # Fixed threshold (default 0.5 — often suboptimal under imbalance)
    pred_fix = (y_score >= threshold_fixed).astype(int)
    out["threshold_fixed"] = float(threshold_fixed)
    out["precision_fixed"] = float(precision_score(y_true, pred_fix, zero_division=0))
    out["recall_fixed"] = float(recall_score(y_true, pred_fix, zero_division=0))
    out["f1_fixed"] = float(f1_score(y_true, pred_fix, zero_division=0))
    out["accuracy_fixed"] = float(accuracy_score(y_true, pred_fix))

    tn, fp, fn, tp = confusion_matrix(y_true, pred_fix, labels=[0, 1]).ravel()
    out["tn_fixed"] = float(tn)
    out["fp_fixed"] = float(fp)
    out["fn_fixed"] = float(fn)
    out["tp_fixed"] = float(tp)

    # Best F1 on PR curve
    best = best_f1_threshold_from_pr_curve(y_true, y_score, eps=eps)
    out["best_threshold_pr_f1"] = float(best["best_threshold"])
    out["precision_at_best_threshold"] = float(best["precision"])
    out["recall_at_best_threshold"] = float(best["recall"])
    out["f1_at_best_threshold"] = float(best["best_f1"])
    out["accuracy_at_best_threshold"] = float(best["accuracy"])

    pred_best = (y_score >= best["best_threshold"]).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, pred_best, labels=[0, 1]).ravel()
    out["tn_best"] = float(tn_b)
    out["fp_best"] = float(fp_b)
    out["fn_best"] = float(fn_b)
    out["tp_best"] = float(tp_b)

    out.update(topk_precision_recall_metrics(y_true, y_score, ks=topk_ks))

    return out


def _fmt_metric(x: Any, decimals: int = 4) -> str:
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return "nan"
        return f"{xf:.{decimals}f}"
    except (TypeError, ValueError):
        return "nan"


def render_training_report_txt(
    history: Dict[str, Any],
    y_true_final: np.ndarray,
    y_pred_final: np.ndarray,
    score_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Full text body for ``training_report.txt`` (single source of truth for file content).

    If ``score_metrics`` is provided (e.g. from ``compute_binary_test_metrics``), writes
    ROC-AUC / PR-AUC, fixed-threshold block, and PR-best-F1-threshold block.
    Otherwise falls back to argmax-only metrics (no PR-AUC).
    """
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true_final, y_pred_final, labels=[0, 1]).ravel()
    specificity_argmax = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0.0

    lines: List[str] = []
    lines.append("=" * 60 + "\n")
    lines.append("Training and Evaluation Report\n")
    lines.append("=" * 60 + "\n\n")

    lines.append("TRAINING HISTORY\n")
    lines.append("-" * 60 + "\n")
    lines.append(f"Number of epochs: {len(history.get('train_loss', []))}\n")
    if "train_loss" in history:
        lines.append(f"Final train loss: {history['train_loss'][-1]:.6f}\n")
    if "val_loss" in history:
        lines.append(f"Final val loss: {history['val_loss'][-1]:.6f}\n")
        lines.append(f"Best val loss: {min(history['val_loss']):.6f}\n")
    lines.append("\n")

    lines.append("TEST SET RESULTS\n")
    lines.append("-" * 60 + "\n")
    if score_metrics is not None:
        lines.append("Threshold-free (ranking quality)\n")
        lines.append(f"  ROC-AUC: {_fmt_metric(score_metrics['roc_auc'])}\n")
        lines.append(f"  PR-AUC:  {_fmt_metric(score_metrics['pr_auc'])}\n")
        lines.append("\n")
        lines.append("At fixed threshold = 0.50 (same as argmax on softmax)\n")
        lines.append(f"  Accuracy:  {_fmt_metric(score_metrics['accuracy_fixed'])}\n")
        lines.append(f"  Precision: {_fmt_metric(score_metrics['precision_fixed'])}\n")
        lines.append(f"  Recall:    {_fmt_metric(score_metrics['recall_fixed'])}\n")
        lines.append(f"  F1:        {_fmt_metric(score_metrics['f1_fixed'])}\n")
        _den = score_metrics["tn_fixed"] + score_metrics["fp_fixed"]
        _spec_f = score_metrics["tn_fixed"] / _den if _den > 0 else 0.0
        lines.append(f"  Specificity (from fixed-threshold CM): {_fmt_metric(_spec_f)}\n")
        lines.append("\n")
        lines.append("At best F1 threshold (from PR curve; recommended under imbalance)\n")
        lines.append(f"  best_threshold: {score_metrics['best_threshold_pr_f1']:.6f}\n")
        lines.append(f"  Accuracy:  {_fmt_metric(score_metrics['accuracy_at_best_threshold'])}\n")
        lines.append(f"  Precision: {_fmt_metric(score_metrics['precision_at_best_threshold'])}\n")
        lines.append(f"  Recall:    {_fmt_metric(score_metrics['recall_at_best_threshold'])}\n")
        lines.append(f"  F1:        {_fmt_metric(score_metrics['f1_at_best_threshold'])}\n")
        lines.append("\n")
        if "precision_at_10" in score_metrics:
            lines.append("Top-K (rank by P(class=1), descending)\n")
            for k in (10, 50, 100):
                pk, rk = f"precision_at_{k}", f"recall_at_{k}"
                if pk in score_metrics and rk in score_metrics:
                    lines.append(
                        f"  Precision@{k}: {_fmt_metric(score_metrics[pk])}  "
                        f"Recall@{k}: {_fmt_metric(score_metrics[rk])}\n"
                    )
            lines.append("\n")
    else:
        lines.append(f"Accuracy:  {accuracy_score(y_true_final, y_pred_final):.4f}\n")
        lines.append(f"Precision: {precision_score(y_true_final, y_pred_final, zero_division=0):.4f}\n")
        lines.append(f"Recall:    {recall_score(y_true_final, y_pred_final, zero_division=0):.4f}\n")
        lines.append(f"F1 Score:  {f1_score(y_true_final, y_pred_final, zero_division=0):.4f}\n")
        lines.append(f"Specificity: {specificity_argmax:.4f}\n")
        lines.append("\n")

    lines.append("CONFUSION MATRIX (argmax / threshold 0.5)\n")
    lines.append("-" * 60 + "\n")
    lines.append(f"True Negatives:  {tn_a}\n")
    lines.append(f"False Positives: {fp_a}\n")
    lines.append(f"False Negatives: {fn_a}\n")
    lines.append(f"True Positives:  {tp_a}\n")
    lines.append("\n")
    if score_metrics is not None:
        lines.append("CONFUSION MATRIX (best F1 threshold on PR curve)\n")
        lines.append("-" * 60 + "\n")
        lines.append(f"True Negatives:  {int(score_metrics['tn_best'])}\n")
        lines.append(f"False Positives: {int(score_metrics['fp_best'])}\n")
        lines.append(f"False Negatives: {int(score_metrics['fn_best'])}\n")
        lines.append(f"True Positives:  {int(score_metrics['tp_best'])}\n")
        lines.append("\n")

    lines.append("CLASS DISTRIBUTION (TEST SET)\n")
    lines.append("-" * 60 + "\n")
    unique, counts = np.unique(y_true_final, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = 100 * count / len(y_true_final)
        lines.append(f"Class {label}: {int(count)} ({percentage:.2f}%)\n")

    return "".join(lines)


__all__ = [
    "compute_metrics",
    "best_f1_threshold_from_pr_curve",
    "compute_binary_test_metrics",
    "topk_precision_recall_metrics",
    "render_training_report_txt",
]
