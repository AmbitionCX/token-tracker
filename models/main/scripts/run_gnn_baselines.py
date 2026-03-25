"""
Train three GNN-related baselines and print test metrics (Table~1 style).

Runs sequentially:
  1. GNN (External only)     — ``baselines.gnn_model.GNNExternalOnly``
  2. GNN + Mean Trace        — ``baselines.gnn_mean_trace_model.GNNMeanTrace``
  3. GNN + Transformer       — ``baselines.gnn_transformer_model.GNNTransformer``

Usage (from ``models/main``)::

    python scripts/run_gnn_baselines.py

Metrics include (threshold 0.5 for Accuracy / P / R / F1): Accuracy, PR-AUC, Precision,
Recall, F1, AUC-ROC, plus **Precision@10/50/100** and **Recall@10/50/100** (top-K by score).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from baselines.gnn_model import GNNExternalOnly
from baselines.gnn_mean_trace_model import GNNMeanTrace
from baselines.gnn_transformer_model import GNNTransformer
from baselines.run_common import load_config, prepare_dataloaders
from training import Trainer, build_train_loss, compute_binary_test_metrics, compute_metrics


def _kwargs_from_config(config: dict) -> Dict[str, Any]:
    mc = config["model"]
    return {
        "trace_encoder_type": mc["trace_encoder"]["type"],
        "trace_hidden_dim": int(mc["trace_encoder"]["hidden_dim"]),
        "trace_num_layers": int(mc["trace_encoder"]["num_layers"]),
        "trace_num_heads": int(mc["trace_encoder"]["num_heads"]),
        "gnn_type": mc["gnn"]["type"],
        "gnn_hidden_dim": int(mc["gnn"]["hidden_dim"]),
        "gnn_num_layers": int(mc["gnn"]["num_layers"]),
        "gnn_num_heads": int(mc["gnn"]["num_heads"]),
        "use_attention": True,
        "attn_num_heads": int(mc["attention"]["num_heads"]),
        "dropout": 0.1,
        "use_trace": True,
        "use_gnn": True,
    }


def _collect_test_scores(
    trainer: Trainer, test_loader, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch_device: Dict[str, Any] = {}
            labels = None
            for k, v in batch.items():
                if k == "labels":
                    labels = v.to(device) if isinstance(v, torch.Tensor) else v
                elif k == "num_edges":
                    continue
                elif isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v
            try:
                logits = trainer.model(**batch_device)
                preds = torch.argmax(logits, dim=-1)
                scores = torch.softmax(logits, dim=-1)[:, 1]
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_scores.append(scores.cpu().numpy())
            except Exception as e:
                print(f"  [warn] batch forward failed: {e}")
                continue
    if not all_labels:
        raise RuntimeError("No test batches produced predictions.")
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_scores),
    )


def _print_six_metrics(title: str, m: Dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  Accuracy:   {m['accuracy_fixed']:.4f}")
    print(f"  PR-AUC:     {m['pr_auc']:.4f}")
    print(f"  Precision:  {m['precision_fixed']:.4f}")
    print(f"  Recall:     {m['recall_fixed']:.4f}")
    print(f"  F1 Score:   {m['f1_fixed']:.4f}")
    print(f"  AUC-ROC:    {m['roc_auc']:.4f}")
    print("  Top-K (rank by P(class=1)):")
    for k in (10, 50, 100):
        print(
            f"    Precision@{k}: {m[f'precision_at_{k}']:.4f}  "
            f"Recall@{k}: {m[f'recall_at_{k}']:.4f}"
        )


BASELINES: List[Tuple[str, str, str, Any]] = [
    ("gnn_external_only", "GNN (External only)", "GNNExternalOnly", GNNExternalOnly),
    ("gnn_mean_trace", "GNN + Mean Trace", "GNNMeanTrace", GNNMeanTrace),
    ("gnn_transformer", "GNN + Transformer", "GNNTransformer", GNNTransformer),
]


def main() -> None:
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    config = load_config(config_path)

    training_cfg = config["training"]
    base_ckpt = Path(training_cfg["checkpoint"]["save_dir"])
    out_dir = base_ckpt / "gnn_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available; using CPU.")

    kw = _kwargs_from_config(config)
    train_loader, val_loader, test_loader = prepare_dataloaders(config, mode="edge")

    loss_fn = build_train_loss(
        loss_name=str(training_cfg.get("loss_fn", "weighted_crossentropy")),
        device=device,
        positive_weight=float(training_cfg.get("positive_weight", 10.0)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        focal_gamma=float(training_cfg.get("focal_gamma", 2.0)),
    )
    metric_for_best = str(training_cfg.get("early_stopping_metric", "macro_f1"))

    rows: List[Dict[str, Any]] = []

    for slug, title, class_name, cls in BASELINES:
        print("\n" + "=" * 70)
        print(f"BASELINE: {title} ({class_name})")
        print("=" * 70)

        model = cls(**kw)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        ckpt_sub = out_dir / slug
        ckpt_sub.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            model=model,
            device=device,
            checkpoint_dir=str(ckpt_sub),
            log_dir=str(ckpt_sub / "logs"),
            use_tensorboard=False,
        )

        _ = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=int(training_cfg["num_epochs"]),
            learning_rate=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
            loss_fn=loss_fn,
            metric_fn=lambda p, y: compute_metrics(p, y),
            patience=int(training_cfg["early_stopping"]["patience"]),
            save_best=training_cfg["checkpoint"]["keep_best"],
            metric_for_best=metric_for_best,
        )

        trainer.save_checkpoint("final_model.pt")
        trainer.load_checkpoint("best_model.pt")

        if test_loader is None:
            print("No test_loader; skipping evaluation.")
            trainer.close()
            continue

        y_true, y_pred, y_scores = _collect_test_scores(trainer, test_loader, device)
        full = compute_binary_test_metrics(y_true, y_scores, threshold_fixed=0.5)

        _print_six_metrics(f"TEST — {title}", full)

        row: Dict[str, Any] = {
            "experiment": slug,
            "title": title,
            "accuracy": float(full["accuracy_fixed"]),
            "pr_auc": float(full["pr_auc"]),
            "precision": float(full["precision_fixed"]),
            "recall": float(full["recall_fixed"]),
            "f1": float(full["f1_fixed"]),
            "auc_roc": float(full["roc_auc"]),
        }
        for k in (10, 50, 100):
            row[f"precision_at_{k}"] = float(full[f"precision_at_{k}"])
            row[f"recall_at_{k}"] = float(full[f"recall_at_{k}"])
        rows.append(row)

        def _jsonable(v: Any) -> Any:
            if isinstance(v, (float, np.floating)):
                x = float(v)
                return None if np.isnan(x) or np.isinf(x) else x
            return float(v) if isinstance(v, (int, np.integer)) else v

        with open(ckpt_sub / "test_metrics.json", "w", encoding="utf-8") as jf:
            json.dump(
                {"experiment": slug, "metrics": {k: _jsonable(v) for k, v in full.items()}},
                jf,
                indent=2,
            )

        trainer.close()

    csv_path = out_dir / "gnn_baselines_metrics.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(
                cf,
                fieldnames=[
                    "experiment",
                    "title",
                    "accuracy",
                    "pr_auc",
                    "precision",
                    "recall",
                    "f1",
                    "auc_roc",
                    "precision_at_10",
                    "recall_at_10",
                    "precision_at_50",
                    "recall_at_50",
                    "precision_at_100",
                    "recall_at_100",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Wrote summary CSV: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
