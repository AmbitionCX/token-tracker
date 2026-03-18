from pipeline.stage1_low_filter import build_step1_dataset
from gates.low_gate import low_gate
from pipeline.stage3_high_filter import run_stage3

def main():
    print("Starting Stage 1: Building dataset with atomic actions...")
    start_block = 10000000
    end_block = 10001000
    TAU_H = 0.9
    dataset=[]

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
            dataset.append(item)  # 收集进入 mid 阶段的交易
            # print(item)  # 只打印进入 mid 阶段的交易


        # print(item)

    # Run stage 3 on the collected mid-stage transactions
    dataset = run_stage3(dataset)

    # Print the final dataset
    for item in dataset:
        print(item)
        
if __name__ == "__main__":
    main()