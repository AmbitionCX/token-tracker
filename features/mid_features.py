# features/mid_features.py

import numpy as np
from collections import Counter
import math


# ----------------------------
# utils
# ----------------------------

def safe_log(x):
    return math.log(x + 1e-9)


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ps = [v / total for v in counter.values() if v > 0]
    return -sum(p * math.log(p) for p in ps)


# ----------------------------
# main feature extractor
# ----------------------------

def extract_mid_features(tx):

    atoms = tx["atomic_actions"]
    summary = tx["summary_counts"]

    # ---------- counts ----------
    num_token_in = summary.get("token_in", 0)
    num_token_out = summary.get("token_out", 0)
    num_approval = summary.get("token_approval", 0)
    num_contract_call = summary.get("contract_call", 0)
    num_delegatecall = summary.get("delegatecall", 0)
    num_multicall = summary.get("multi_call", 0)

    # ---------- assets ----------
    tokens = [a["token"] for a in atoms if "token" in a]
    token_counter = Counter(tokens)

    unique_assets = len(token_counter)
    total_token_events = sum(token_counter.values())

    top_token_share = 0.0
    if total_token_events > 0:
        top_token_share = max(token_counter.values()) / total_token_events

    asset_entropy = entropy(token_counter)

    # ---------- flow（⚠️ 先用近似版本） ----------
    # 你后面可以换成 USD 精确值

    sum_in = num_token_in
    sum_out = num_token_out

    sum_in_log = safe_log(sum_in)
    sum_out_log = safe_log(sum_out)

    net_flow = sum_in - sum_out

    in_out_ratio = sum_in / (sum_out + 1e-6)

    abs_flow = sum_in + sum_out

    flow_imbalance = abs(sum_in - sum_out) / (abs_flow + 1e-6)

    # ---------- interaction ----------
    # ⚠️ 目前用近似（等你 trace 再升级）

    max_call_depth = tx.get("max_call_depth", 0)
    num_internal_calls = tx.get("num_internal_calls", 0)

    if num_contract_call == 0:
        call_pattern_type = 0
    elif num_contract_call == 1:
        call_pattern_type = 1
    else:
        call_pattern_type = 2

    # ---------- flags ----------
    approval_then_transfer_flag = int(
        num_approval > 0 and num_token_out > 0
    )

    has_delegatecall = int(num_delegatecall > 0)
    has_contract_creation = int(summary.get("contract_creation", 0) > 0)
    is_multicall = int(num_multicall > 0)

    # ---------- final vector ----------

    feats = np.array([
        # counts
        num_token_in,
        num_token_out,
        num_approval,
        num_contract_call,
        num_delegatecall,
        num_multicall,

        # assets
        unique_assets,
        top_token_share,
        asset_entropy,

        # flow
        sum_in_log,
        sum_out_log,
        net_flow,
        in_out_ratio,
        abs_flow,
        flow_imbalance,

        # interaction
        max_call_depth,
        num_internal_calls,
        call_pattern_type,

        # flags
        approval_then_transfer_flag,
        has_delegatecall,
        has_contract_creation,
        is_multicall
    ], dtype=float)

    return feats