from pipeline.stage1_low import build_step1_dataset
from pipeline.stage2_low_mid import run_stage2
from pipeline.stage3_mid import run_stage3

def main():
    print("Starting Stage 1: Building dataset with atomic actions...")
    start_block = 10000000
    end_block = 10001000
    TAU_H = 0.9

    # Step 1: Build initial dataset
    dataset = list(build_step1_dataset(start_block, end_block))

    # Step 2: Apply low-mid gate to filter transactions
    dataset = run_stage2(dataset, tau_H=TAU_H)
    
    # Step 3: Run mid-level clustering on mid-stage transactions
    dataset = run_stage3(dataset)   

    # Print the final dataset
    for item in dataset:
        print(item)
        
if __name__ == "__main__":
    main()