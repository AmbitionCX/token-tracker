#!/usr/bin/env python3
"""
Train Baselines: Run all 4 baseline models and report metrics.

Baselines:
  1. XGBoost (External only)        — sklearn-style, uses 4 external features
  2. MLP (External + Mean Trace)    — PyTorch, uses external + mean-pooled trace
  3. LSTM (Trace only)              — PyTorch, sequence model on trace
  4. Transformer (Trace only)       — PyTorch, transformer on trace

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --models xgboost mlp
    python scripts/train_baselines.py --device cpu
"""

import os
import sys
import json
import time
import argparse
import pickle
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve project root so imports work when running from <project>/scripts/
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.metrics import topk_precision_recall_metrics

from data.data_loader import TransactionDataLoader, CachedDataLoader
from core.edge_feature_extractor import (
    ExternalTransactionFeatures,
    InternalCallTraceFeatures,
    EdgeFeatureMask,
    CallEventEmbedding,
)
from baselines.xgboost_model import XGBoostBaseline
from baselines.mlp_model import MLPBaseline
from baselines.lstm_model import LSTMBaseline
from baselines.transformer_model import TransformerBaseline


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Data
    "table_name": "tx_joined_4000000_4010000",
    "start_block": 4000000,
    "end_block": 4010000,
    "chunk_size": 5000,
    "cache_dir": "./cache",

    # Data split ratios (time-series split)
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # Trace config
    "max_trace_length": 256,
    "trace_embedding_dim": 128,

    # Training (PyTorch models)
    "batch_size": 256,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,
    "positive_weight": "auto",  # Will be computed from actual pos/neg ratio

    # General
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ============================================================================
# Data Loading & Feature Extraction
# ============================================================================

def load_and_extract_features(config: dict) -> Dict[str, np.ndarray]:
    """
    Stream transactions from PostgreSQL, extract features, and cache them.

    Returns a dict with numpy arrays:
        external_features : (N, 4)
        call_type_ids     : (N, max_seq_len)
        contract_ids      : (N, max_seq_len)
        func_selector_ids : (N, max_seq_len)
        depths            : (N, max_seq_len)
        status_ids        : (N, max_seq_len)
        input_sizes       : (N, max_seq_len)
        output_sizes      : (N, max_seq_len)
        gas_vals          : (N, max_seq_len)
        trace_mask        : (N, max_seq_len)  bool
        labels            : (N,)              int
    """
    cache = CachedDataLoader(cache_dir=config["cache_dir"])
    cache_name = f"baseline_features_{config['start_block']}_{config['end_block']}"

    # Try loading from cache first
    cached = cache.load_cache(cache_name)
    if cached is not None:
        print(f"[Data] Loaded cached features ({cached['labels'].shape[0]} samples)")
        return cached

    print("[Data] No cache found — streaming from PostgreSQL …")
    loader = TransactionDataLoader(table_name=config["table_name"])
    ext_extractor = ExternalTransactionFeatures()
    trace_extractor = InternalCallTraceFeatures(
        max_sequence_length=config["max_trace_length"]
    )

    # Accumulators
    all_external = []
    all_call_type = []
    all_contract = []
    all_func_sel = []
    all_depth = []
    all_status = []
    all_input_sz = []
    all_output_sz = []
    all_gas = []
    all_mask = []
    all_labels = []

    for chunk_df in loader.stream_transactions(
        config["start_block"], config["end_block"], config["chunk_size"]
    ):
        for _, row in tqdm(
            chunk_df.iterrows(),
            total=len(chunk_df),
            desc="Extracting features",
            leave=False,
        ):
            tx = row.to_dict()

            # External features (4 dims)
            ext_feat = ext_extractor.extract(tx)
            all_external.append(ext_feat)

            # Internal trace features (8 arrays)
            trace_data = tx.get("trace_data", {})
            (ct, ci, fs, dp, st, ins, outs, gv) = trace_extractor.extract(trace_data)

            # Pad / truncate to max_trace_length
            max_len = config["max_trace_length"]
            ct = _pad_or_truncate(ct, max_len, dtype=np.int32)
            ci = _pad_or_truncate(ci, max_len, dtype=np.int32)
            fs = _pad_or_truncate(fs, max_len, dtype=np.int32)
            dp = _pad_or_truncate(dp, max_len, dtype=np.int32)
            st = _pad_or_truncate(st, max_len, dtype=np.int32)
            ins = _pad_or_truncate(ins, max_len, dtype=np.float32)
            outs = _pad_or_truncate(outs, max_len, dtype=np.float32)
            gv = _pad_or_truncate(gv, max_len, dtype=np.float32)

            mask = EdgeFeatureMask.create_mask(ct)

            all_call_type.append(ct)
            all_contract.append(ci)
            all_func_sel.append(fs)
            all_depth.append(dp)
            all_status.append(st)
            all_input_sz.append(ins)
            all_output_sz.append(outs)
            all_gas.append(gv)
            all_mask.append(mask)

            all_labels.append(int(tx.get("is_suspicious", 0)))

    data = {
        "external_features": np.stack(all_external),
        "call_type_ids": np.stack(all_call_type),
        "contract_ids": np.stack(all_contract),
        "func_selector_ids": np.stack(all_func_sel),
        "depths": np.stack(all_depth),
        "status_ids": np.stack(all_status),
        "input_sizes": np.stack(all_input_sz),
        "output_sizes": np.stack(all_output_sz),
        "gas_vals": np.stack(all_gas),
        "trace_mask": np.stack(all_mask),
        "labels": np.array(all_labels, dtype=np.int64),
    }

    cache.save_cache(cache_name, data)
    print(f"[Data] Extracted & cached {data['labels'].shape[0]} samples")
    return data


def _pad_or_truncate(arr: np.ndarray, length: int, dtype=np.float32) -> np.ndarray:
    """Pad or truncate a 1-D array to exactly `length`."""
    if len(arr) >= length:
        return arr[:length].astype(dtype)
    padded = np.zeros(length, dtype=dtype)
    padded[: len(arr)] = arr
    return padded


def stratified_split(
    data: Dict[str, np.ndarray],
    train_ratio: float,
    val_ratio: float,
    random_state: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """
    Stratified split: ensure positive samples appear in all splits.

    With extreme imbalance (e.g. 96 positives out of 105k), a pure time-series
    split can leave zero positives in val or test.  We use sklearn's
    StratifiedShuffleSplit to guarantee proportional representation.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = data["labels"]
    n = len(labels)

    # First split: train vs (val+test)
    test_val_ratio = 1.0 - train_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=random_state)
    train_idx, valtest_idx = next(sss1.split(np.zeros(n), labels))

    # Second split: val vs test (within the val+test portion)
    val_fraction_of_remainder = val_ratio / test_val_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction_of_remainder, random_state=random_state)
    valtest_labels = labels[valtest_idx]
    val_local_idx, test_local_idx = next(sss2.split(np.zeros(len(valtest_idx)), valtest_labels))
    val_idx = valtest_idx[val_local_idx]
    test_idx = valtest_idx[test_local_idx]

    def _index(d, idx):
        return {k: v[idx] for k, v in d.items()}

    train_data = _index(data, train_idx)
    val_data = _index(data, val_idx)
    test_data = _index(data, test_idx)

    # Report positive counts per split
    for name, d in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        pos = int(d["labels"].sum())
        total = len(d["labels"])
        print(f"  {name}: {total} samples, {pos} positive ({pos/total*100:.2f}%)")

    return train_data, val_data, test_data


# ============================================================================
# PyTorch Dataset for trace-based models (MLP / LSTM / Transformer)
# ============================================================================

class TraceDataset(Dataset):
    """PyTorch Dataset that wraps the extracted feature arrays."""

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.n = data["labels"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.data.items()}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack list of sample dicts into a batched dict."""
    keys = batch[0].keys()
    return {k: torch.stack([s[k] for s in batch]) for k in keys}


# ============================================================================
# Metric helpers
# ============================================================================

def compute_6_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute the 6 primary metrics plus Top-K Precision@K / Recall@K (K=10,50,100)."""
    out: Dict[str, float] = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "PR_AUC": average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC_ROC": roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else 0.0,
    }
    tk = topk_precision_recall_metrics(y_true, y_prob, ks=(10, 50, 100))
    for k in (10, 50, 100):
        out[f"Precision@{k}"] = tk[f"precision_at_{k}"]
        out[f"Recall@{k}"] = tk[f"recall_at_{k}"]
    return out


def print_metrics(name: str, metrics: Dict[str, float]):
    """Pretty-print metrics for a model."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<18s}: {v:.4f}")
    print(f"{'='*60}")


# ============================================================================
# Mean trace pooling helper (for MLP baseline)
# ============================================================================

def compute_mean_trace_embeddings(
    data: Dict[str, np.ndarray],
    call_event_emb: nn.Module,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Run CallEventEmbedding and mean-pool to get (N, trace_dim) features.

    This converts the raw tokenised trace into a fixed-length vector
    that can be fed to the MLP baseline.
    """
    dataset = TraceDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    call_event_emb = call_event_emb.to(device).eval()
    all_trace_vecs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing mean trace embeddings"):
            emb = call_event_emb(
                batch["call_type_ids"].to(device),
                batch["contract_ids"].to(device),
                batch["func_selector_ids"].to(device),
                batch["depths"].to(device),
                batch["status_ids"].to(device),
                batch["input_sizes"].float().to(device),
                batch["output_sizes"].float().to(device),
                batch["gas_vals"].float().to(device),
                batch["trace_mask"].to(device),
            )  # (B, L, emb_dim)

            mask = batch["trace_mask"].unsqueeze(-1).float().to(device)  # (B, L, 1)
            pooled = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # (B, emb_dim)
            all_trace_vecs.append(pooled.cpu().numpy())

    return np.concatenate(all_trace_vecs, axis=0)


# ============================================================================
# Training loop for PyTorch baselines (MLP / LSTM / Transformer)
# ============================================================================

def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    model_name: str,
    prepare_batch_fn=None,
) -> nn.Module:
    """
    Generic training loop for any PyTorch baseline.

    Args:
        model: the nn.Module to train
        train_loader / val_loader: DataLoaders
        config: global config dict
        model_name: for logging
        prepare_batch_fn: callable(batch, device) → (inputs_tuple, labels)
    """
    device = torch.device(config["device"])
    model = model.to(device)

    pos_weight = torch.tensor(
        [1.0, config["positive_weight"]], dtype=torch.float32
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1} Train", leave=False):
            inputs, labels = prepare_batch_fn(batch, device)
            optimizer.zero_grad()
            logits = model(*inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # ---- Validate ----
        val_metrics, _ = evaluate_pytorch_model(model, val_loader, device, prepare_batch_fn)
        val_f1 = val_metrics["F1_Score"]

        print(
            f"  [{model_name}] Epoch {epoch+1:>2d}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_F1={val_f1:.4f}  val_AUC={val_metrics['AUC_ROC']:.4f}"
        )

        # Early stopping on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_f1 >= 0:
        model.load_state_dict(best_state)
    return model


def evaluate_pytorch_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    prepare_batch_fn,
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate a PyTorch model and return 6 metrics + raw predictions."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = prepare_batch_fn(batch, device)
            logits = model(*inputs)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    metrics = compute_6_metrics(y_true, y_pred, y_prob)
    return metrics, (y_true, y_pred, y_prob)


# ============================================================================
# Per-model batch preparation functions
# ============================================================================

def _prepare_mlp_batch(batch, device):
    """MLP uses pre-computed external + mean trace features."""
    x = batch["features"].float().to(device)
    # Split external (first 4) and trace (remaining)
    ext = x[:, :4]
    trace = x[:, 4:]
    labels = batch["labels"].long().to(device)
    return (ext, trace), labels


def _make_trace_batch_preparer(call_event_emb: nn.Module):
    """Factory for LSTM/Transformer batch preparers that embed on-the-fly."""

    def _prepare(batch, device):
        call_event_emb_dev = call_event_emb.to(device)
        call_event_emb_dev.eval()

        with torch.no_grad():
            emb = call_event_emb_dev(
                batch["call_type_ids"].to(device),
                batch["contract_ids"].to(device),
                batch["func_selector_ids"].to(device),
                batch["depths"].to(device),
                batch["status_ids"].to(device),
                batch["input_sizes"].float().to(device),
                batch["output_sizes"].float().to(device),
                batch["gas_vals"].float().to(device),
                batch["trace_mask"].to(device),
            )  # (B, L, emb_dim)

        mask = batch["trace_mask"].to(device)
        labels = batch["labels"].long().to(device)
        return (emb, mask), labels

    return _prepare


# ============================================================================
# MLP-specific dataset (pre-computed features)
# ============================================================================

class PrecomputedDataset(Dataset):
    """Dataset that holds pre-computed feature vectors + labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "mlp", "lstm", "transformer"],
        choices=["xgboost", "mlp", "lstm", "transformer"],
        help="Which baselines to run",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size

    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["random_seed"])

    print(f"Device: {config['device']}")
    print(f"Models to run: {args.models}")

    # ------------------------------------------------------------------
    # 1. Load data & extract features
    # ------------------------------------------------------------------
    data = load_and_extract_features(config)
    n_total = data["labels"].shape[0]
    n_pos = int(data["labels"].sum())
    n_neg = n_total - n_pos
    print(f"\n[Data] Total: {n_total}  |  Positive: {n_pos}  |  Negative: {n_neg}")
    print(f"[Data] Positive ratio: {n_pos / n_total:.4f}")

    # Auto-compute positive_weight from class imbalance ratio
    if config["positive_weight"] == "auto":
        config["positive_weight"] = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        print(f"[Data] Auto positive_weight: {config['positive_weight']:.1f}")

    # ------------------------------------------------------------------
    # 2. Stratified split (ensures positives in every split)
    # ------------------------------------------------------------------
    train_data, val_data, test_data = stratified_split(
        data, config["train_ratio"], config["val_ratio"],
        random_state=config["random_seed"],
    )
    print(
        f"[Split] Train: {train_data['labels'].shape[0]}  "
        f"Val: {val_data['labels'].shape[0]}  "
        f"Test: {test_data['labels'].shape[0]}"
    )

    # Shared CallEventEmbedding (for MLP mean trace + LSTM/Transformer)
    call_event_emb = CallEventEmbedding(
        call_type_vocab_size=10,
        contract_vocab_size=50000,
        func_selector_vocab_size=100000,
        depth_max=50,
        embedding_dim=config["trace_embedding_dim"] // 8,
    )
    emb_dim = call_event_emb.output_dim  # actual per-token embedding dim

    results = {}

    # ==================================================================
    # Baseline 1: XGBoost (External only)
    # ==================================================================
    if "xgboost" in args.models:
        print("\n" + "=" * 60)
        print("  Training Baseline 1: XGBoost (External only)")
        print("=" * 60)
        t0 = time.time()

        pos_neg_ratio = n_neg / max(n_pos, 1)
        xgb = XGBoostBaseline(pos_neg_ratio=pos_neg_ratio, random_state=config["random_seed"])
        xgb.train(
            train_data["external_features"],
            train_data["labels"],
            val_data["external_features"],
            val_data["labels"],
        )
        metrics = xgb.evaluate(test_data["external_features"], test_data["labels"])
        results["XGBoost (External only)"] = metrics
        print_metrics("XGBoost (External only)", metrics)
        print(f"  Time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Baseline 2: MLP (External + Mean Trace)
    # ==================================================================
    if "mlp" in args.models:
        print("\n" + "=" * 60)
        print("  Training Baseline 2: MLP (External + Mean Trace)")
        print("=" * 60)
        t0 = time.time()

        # Pre-compute mean trace embeddings for all splits
        print("  Computing mean trace embeddings …")
        train_trace = compute_mean_trace_embeddings(train_data, call_event_emb, config["device"], config["batch_size"])
        val_trace = compute_mean_trace_embeddings(val_data, call_event_emb, config["device"], config["batch_size"])
        test_trace = compute_mean_trace_embeddings(test_data, call_event_emb, config["device"], config["batch_size"])

        # Concatenate external + mean trace
        train_feats = np.concatenate([train_data["external_features"], train_trace], axis=1)
        val_feats = np.concatenate([val_data["external_features"], val_trace], axis=1)
        test_feats = np.concatenate([test_data["external_features"], test_trace], axis=1)

        input_dim = train_feats.shape[1]  # 4 + emb_dim

        mlp_train_ds = PrecomputedDataset(train_feats, train_data["labels"])
        mlp_val_ds = PrecomputedDataset(val_feats, val_data["labels"])
        mlp_test_ds = PrecomputedDataset(test_feats, test_data["labels"])

        mlp_train_loader = DataLoader(mlp_train_ds, batch_size=config["batch_size"], shuffle=True)
        mlp_val_loader = DataLoader(mlp_val_ds, batch_size=config["batch_size"], shuffle=False)
        mlp_test_loader = DataLoader(mlp_test_ds, batch_size=config["batch_size"], shuffle=False)

        mlp_model = MLPBaseline(input_dim=input_dim, hidden_dim=128, num_classes=2, dropout=0.3)
        mlp_model = train_pytorch_model(
            mlp_model, mlp_train_loader, mlp_val_loader, config,
            model_name="MLP", prepare_batch_fn=_prepare_mlp_batch,
        )
        metrics, _ = evaluate_pytorch_model(mlp_model, mlp_test_loader, config["device"], _prepare_mlp_batch)
        results["MLP (External + Mean Trace)"] = metrics
        print_metrics("MLP (External + Mean Trace)", metrics)
        print(f"  Time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Baseline 3: LSTM (Trace only)
    # ==================================================================
    if "lstm" in args.models:
        print("\n" + "=" * 60)
        print("  Training Baseline 3: LSTM (Trace only)")
        print("=" * 60)
        t0 = time.time()

        lstm_train_loader = DataLoader(
            TraceDataset(train_data), batch_size=config["batch_size"],
            shuffle=True, collate_fn=collate_fn,
        )
        lstm_val_loader = DataLoader(
            TraceDataset(val_data), batch_size=config["batch_size"],
            shuffle=False, collate_fn=collate_fn,
        )
        lstm_test_loader = DataLoader(
            TraceDataset(test_data), batch_size=config["batch_size"],
            shuffle=False, collate_fn=collate_fn,
        )

        prepare_trace = _make_trace_batch_preparer(call_event_emb)
        lstm_model = LSTMBaseline(input_dim=emb_dim, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.1)
        lstm_model = train_pytorch_model(
            lstm_model, lstm_train_loader, lstm_val_loader, config,
            model_name="LSTM", prepare_batch_fn=prepare_trace,
        )
        metrics, _ = evaluate_pytorch_model(lstm_model, lstm_test_loader, config["device"], prepare_trace)
        results["LSTM (Trace only)"] = metrics
        print_metrics("LSTM (Trace only)", metrics)
        print(f"  Time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Baseline 4: Transformer (Trace only)
    # ==================================================================
    if "transformer" in args.models:
        print("\n" + "=" * 60)
        print("  Training Baseline 4: Transformer (Trace only)")
        print("=" * 60)
        t0 = time.time()

        tf_train_loader = DataLoader(
            TraceDataset(train_data), batch_size=config["batch_size"],
            shuffle=True, collate_fn=collate_fn,
        )
        tf_val_loader = DataLoader(
            TraceDataset(val_data), batch_size=config["batch_size"],
            shuffle=False, collate_fn=collate_fn,
        )
        tf_test_loader = DataLoader(
            TraceDataset(test_data), batch_size=config["batch_size"],
            shuffle=False, collate_fn=collate_fn,
        )

        prepare_trace = _make_trace_batch_preparer(call_event_emb)
        tf_model = TransformerBaseline(
            input_dim=emb_dim, hidden_dim=128, num_layers=2, num_heads=4,
            num_classes=2, dropout=0.1, max_seq_len=config["max_trace_length"],
        )
        tf_model = train_pytorch_model(
            tf_model, tf_train_loader, tf_val_loader, config,
            model_name="Transformer", prepare_batch_fn=prepare_trace,
        )
        metrics, _ = evaluate_pytorch_model(tf_model, tf_test_loader, config["device"], prepare_trace)
        results["Transformer (Trace only)"] = metrics
        print_metrics("Transformer (Trace only)", metrics)
        print(f"  Time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Summary Table
    # ==================================================================
    if results:
        print("\n\n")
        print("=" * 90)
        print("  BASELINE RESULTS SUMMARY")
        print("=" * 90)
        header = f"{'Model':<35s} {'Acc':>7s} {'PR_AUC':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUC':>7s}"
        print(header)
        print("-" * 90)
        for name, m in results.items():
            row = (
                f"{name:<35s} "
                f"{m['Accuracy']:>7.4f} "
                f"{m['PR_AUC']:>7.4f} "
                f"{m['Precision']:>7.4f} "
                f"{m['Recall']:>7.4f} "
                f"{m['F1_Score']:>7.4f} "
                f"{m['AUC_ROC']:>7.4f}"
            )
            print(row)
        print("=" * 90)

        # Save results to JSON
        out_path = PROJECT_ROOT / "baselines" / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()