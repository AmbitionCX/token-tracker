# gates/low_gate.py

import math

ATOM_TYPES = [
    "token_in",
    "token_out",
    "token_approval",
    "contract_call",
    "multi_call",
    "delegatecall",
    "contract_creation"
]


def compute_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0

    ps = [v / total for v in counts.values() if v > 0]
    return -sum(p * math.log(p) for p in ps)


def derive_low_intent(summary):

    if not summary:
        return "unknown_low"

    if summary.get("contract_creation", 0) > 0:
        return "contract_creation"

    if summary.get("token_approval", 0) > 0 \
       and summary.get("contract_call", 0) == 0 \
       and summary.get("token_in", 0) == 0 \
       and summary.get("token_out", 0) == 0:
        return "token_approval"

    if summary.get("delegatecall", 0) > 0:
        return "delegatecall_interaction"

    if summary.get("multi_call", 0) > 0 \
       or summary.get("contract_call", 0) >= 2:
        return "multi_call_interaction"

    total_asset = summary.get("token_in", 0) + summary.get("token_out", 0)

    if summary.get("contract_call", 0) == 0:

        if total_asset == 1:
            return "simple_transfer"

        if total_asset > 1:
            return "asset_only_complex"

    if summary.get("contract_call", 0) == 1:
        return "contract_interaction"

    return "unknown_low"


def low_gate(summary_counts, unique_assets, tau_H=0.9):

    counts = {k: summary_counts.get(k, 0) for k in ATOM_TYPES}

    total = sum(counts.values())
    D_asset = len(unique_assets)

    # ---------- rule-based early exit ----------

    if (
        D_asset == 1
        and total <= 2
        and counts["contract_call"] == 0
        and counts["multi_call"] == 0
    ):
        return {
            "decision": "stop",
            "intent": "simple_transfer"
        }

    if (
        counts["token_approval"] >= 1
        and total == counts["token_approval"]
    ):
        return {
            "decision": "stop",
            "intent": "token_approval"
        }

    # ---------- entropy ----------

    H = compute_entropy(counts)

    if H > tau_H:
        return {
            "decision": "go_mid",
            "H_low": H
        }

    else:
        return {
            "decision": "stop",
            "intent": derive_low_intent(summary_counts),
            "H_low": H
        }