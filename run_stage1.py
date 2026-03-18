from pipeline.stage1_low_filter import build_step1_dataset
from gates.low_gate import low_gate
def main():
    print("Starting Stage 1: Building dataset with atomic actions...")
    start_block = 10000000
    end_block = 10000010
    TAU_H = 0.9

    for item in build_step1_dataset(start_block, end_block):
        # print(item)
        gate_out = low_gate(
            item["summary_counts"],
            item["unique_assets"],
            TAU_H
        )

        if gate_out["decision"] == "stop":

            item["stage"] = "low"
            item["low_intent"] = gate_out["intent"]

        else:

            item["stage"] = "mid"
            item["H_low"] = gate_out["H_low"]
            print(item)  # 只打印进入 mid 阶段的交易


        # print(item)

if __name__ == "__main__":
    main()