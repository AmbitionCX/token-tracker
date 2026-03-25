"""Shared config loading, graph construction, and DataLoaders for Table 1 scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

# models/main as cwd
_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = str(_ROOT / "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataloaders(config: dict, mode: str = "edge") -> Tuple[Any, Any, Any]:
    """
    Build temporal graphs and return (train_loader, val_loader, test_loader).

    Args:
        mode: ``edge`` → EdgeLevelDataset; ``graph`` → GraphWindowDataset (batch_size=1 windows).
    """
    import sys

    sys.path.insert(0, str(_ROOT))
    from core import EdgeFeatureExtractor
    from data import TransactionDataLoader, GraphConstructor, GraphDataLoader

    data_cfg = config["data"]
    db_loader = TransactionDataLoader(table_name=data_cfg["table_name"])

    feature_extractor = EdgeFeatureExtractor(
        trace_embedding_dim=int(config["model"]["trace_encoder"]["hidden_dim"]),
        max_trace_length=256,
        use_trace=True,
        use_external=True,
    )

    graph_constructor = GraphConstructor(
        data_loader=db_loader,
        feature_extractor=feature_extractor,
        temporal_window=int(data_cfg.get("temporal_window", data_cfg.get("window_size", 1000))),
        cache_dir=data_cfg.get("cache_dir", "./cache"),
    )

    graphs = graph_constructor.construct_graphs(
        start_block=int(data_cfg["start_block"]),
        end_block=int(data_cfg["end_block"]),
        use_cache=bool(data_cfg.get("cache_enabled", True)),
        force_rebuild=False,
    )

    graph_loader = GraphDataLoader(
        graphs=graphs,
        batch_size=int(data_cfg.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    return graph_loader.create_dataloaders(
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        test_ratio=1.0 - float(data_cfg["train_ratio"]) - float(data_cfg["val_ratio"]),
        mode=mode,
    )


def graphs_from_config(config: dict) -> List[dict]:
    """Load or build temporal graphs (same as training) for XGBoost feature extraction."""
    import sys

    sys.path.insert(0, str(_ROOT))
    from core import EdgeFeatureExtractor
    from data import TransactionDataLoader, GraphConstructor

    data_cfg = config["data"]
    db_loader = TransactionDataLoader(table_name=data_cfg["table_name"])
    feature_extractor = EdgeFeatureExtractor(
        trace_embedding_dim=int(config["model"]["trace_encoder"]["hidden_dim"]),
        max_trace_length=256,
        use_trace=True,
        use_external=True,
    )
    graph_constructor = GraphConstructor(
        data_loader=db_loader,
        feature_extractor=feature_extractor,
        temporal_window=int(data_cfg.get("temporal_window", data_cfg.get("window_size", 1000))),
        cache_dir=data_cfg.get("cache_dir", "./cache"),
    )
    return graph_constructor.construct_graphs(
        start_block=int(data_cfg["start_block"]),
        end_block=int(data_cfg["end_block"]),
        use_cache=bool(data_cfg.get("cache_enabled", True)),
        force_rebuild=False,
    )


def stack_external_labels(graphs: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack 4-D external features and edge labels from all graphs."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for g in graphs:
        for i in range(int(g["num_edges"])):
            ef = g["edge_features"][i]
            ext = np.asarray(ef["external_features"], dtype=np.float32).reshape(-1)
            if ext.size >= 4:
                ext = ext[:4]
            else:
                pad = np.zeros(4, dtype=np.float32)
                pad[: ext.size] = ext
                ext = pad
            xs.append(ext)
            ys.append(int(g["edge_labels"][i]))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)
