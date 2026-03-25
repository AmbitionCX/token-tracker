"""
Loss functions for imbalanced binary classification (e.g. rare positive / suspicious class).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional


def binary_class_weighted_cross_entropy(
    positive_weight: float,
    device: Union[torch.device, str],
    *,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """
    Cross-entropy with per-class weights: class 0 (negative) = 1.0, class 1 (positive) = ``positive_weight``.

    When positives are very rare, increase ``positive_weight`` so the model pays more attention to
    missing positives (often 5–100 depending on the imbalance ratio; try ``sqrt(N_neg/N_pos)`` as a start).

    Args:
        positive_weight: Multiplier for the positive class in the loss.
        device: Device for the weight tensor (match model / batch device).
        label_smoothing: Optional label smoothing (0 = off).
    """
    w = torch.tensor([1.0, float(positive_weight)], device=device)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=float(label_smoothing))


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification (logits -> softmax).
    CE term uses optional class ``weight`` (same role as weighted cross-entropy).
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("class_weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
        )
        pt = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def build_train_loss(
    *,
    loss_name: str,
    device: Union[torch.device, str],
    positive_weight: float = 10.0,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Build training loss from ``config['training']['loss_fn']``.

    - ``crossentropy`` / ``ce``: unweighted (usually poor on heavy imbalance).
    - ``weighted_crossentropy`` (recommended): class-weighted CE for labels 0/1.
    - ``focal`` / ``focal_loss``: focal loss with class weights (same ``positive_weight`` as WCE).
    """
    name = (loss_name or "weighted_crossentropy").lower().replace("-", "_")
    if name in ("crossentropy", "ce"):
        return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    if name in ("weighted_crossentropy", "weighted_ce", "wce"):
        return binary_class_weighted_cross_entropy(
            positive_weight, device, label_smoothing=label_smoothing
        )
    if name in ("focal", "focal_loss"):
        w = torch.tensor([1.0, float(positive_weight)], device=device)
        return BinaryFocalLoss(gamma=focal_gamma, weight=w, label_smoothing=label_smoothing)
    raise ValueError(
        f"Unknown training.loss_fn: {loss_name!r}. "
        "Use 'crossentropy', 'weighted_crossentropy', or 'focal'."
    )


__all__ = [
    "binary_class_weighted_cross_entropy",
    "BinaryFocalLoss",
    "build_train_loss",
]
