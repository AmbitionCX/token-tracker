# pipeline/stage3_high_filter.py

import numpy as np
from features.mid_features import extract_mid_features
from models.clustering import MidClusteringModel


def run_stage3(dataset):

    # 只取 mid 的
    mid_data = [tx for tx in dataset if tx.get("stage") == "mid"]

    if len(mid_data) == 0:
        return dataset

    # ---------- feature extraction ----------
    X = []

    for tx in mid_data:
        feats = extract_mid_features(tx)
        tx["feat_summary"] = feats
        X.append(feats)

    X = np.vstack(X)

    # ---------- clustering ----------
    model = MidClusteringModel(n_clusters=8)
    model.fit(X)

    clusters, confs = model.predict(X)

    # ---------- visualize clustering ----------
    model.visualize(X, clusters, save_path='cluster_visualization.png')

    # ---------- assign ----------
    for i, tx in enumerate(mid_data):
        tx["mid_cluster"] = int(clusters[i])
        tx["cluster_conf"] = float(confs[i])
        tx["z_t"] = model.transform([tx["feat_summary"]])[0]

    return dataset