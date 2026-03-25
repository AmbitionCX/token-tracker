"""
Mine false-positive edges on the **training** split (pred positive, label negative).

Writes a JSON list of ``EdgeLevelDataset`` indices for use with
``data.hard_negative_indices_file`` + ``hard_negative_boost`` in ``config.yaml``.

Usage (from ``models/main``, after you have a checkpoint)::

    python scripts/mine_hard_negatives.py ./checkpoints/best_model.pt

Optional::

    python scripts/mine_hard_negatives.py ./checkpoints/best_model.pt --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from baselines.run_common import load_config, prepare_dataloaders
from models import SequenceGNNModel


def _build_model(config: dict) -> torch.nn.Module:
    mc = config["model"]
    return SequenceGNNModel(
        trace_encoder_type=mc["trace_encoder"]["type"],
        trace_hidden_dim=int(mc["trace_encoder"]["hidden_dim"]),
        trace_num_layers=int(mc["trace_encoder"]["num_layers"]),
        trace_num_heads=int(mc["trace_encoder"]["num_heads"]),
        gnn_type=mc["gnn"]["type"],
        gnn_hidden_dim=int(mc["gnn"]["hidden_dim"]),
        gnn_num_layers=int(mc["gnn"]["num_layers"]),
        gnn_num_heads=int(mc["gnn"]["num_heads"]),
        use_attention=True,
        attn_num_heads=int(mc["attention"]["num_heads"]),
        dropout=0.1,
        use_trace=True,
        use_gnn=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Positive probability threshold for predicted positive",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: checkpoints/hard_negative_indices.json)",
    )
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _, _ = prepare_dataloaders(config, mode="edge")
    ds = train_loader.dataset

    model = _build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint must contain 'model_state' or 'model_state_dict'")
    model.load_state_dict(state)
    model.eval()

    hard: list[int] = []
    thr = float(args.threshold)

    with torch.no_grad():
        for i in range(len(ds)):
            batch = default_collate([ds[i]])
            labels = batch.pop("labels").to(device)
            batch.pop("num_edges", None)
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(**batch_device)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            pred = 1 if prob >= thr else 0
            lab = int(labels.item())
            if pred == 1 and lab == 0:
                hard.append(i)

    out_path = Path(args.out or (config["training"]["checkpoint"]["save_dir"] + "/hard_negative_indices.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hard, f, indent=2)

    print(f"Mined {len(hard)} hard negatives (FP at thr={thr}) → {out_path}")


if __name__ == "__main__":
    main()
