# pipeline/stage2_mid_filter.py

from gates.low_mid_gate import low_gate


def run_stage2(dataset, tau_H=0.9):

    results = []

    for tx in dataset:

        gate_out = low_gate(
            tx["summary_counts"],
            tx["unique_assets"],
            tau_H
        )

        if gate_out["decision"] == "stop":

            tx["stage"] = "low"
            tx["low_intent"] = gate_out["intent"]

        else:

            tx["stage"] = "mid"
            tx["H_low"] = gate_out["H_low"]

        results.append(tx)

    return results