# pipeline/stage1_low.py
from data.load_tx import stream_transactions
from features.atom_extractor import extract_atomic_actions


def build_step1_dataset(start_block, end_block):

    for chunk in stream_transactions(start_block, end_block):

        for tx in chunk.to_dict("records"):

            atoms, summary_counts, unique_assets = extract_atomic_actions(tx)
            
            if not atoms:  
                continue  # 跳过没有 token 行为的交易

            yield {

                "tx_hash": tx["transaction_hash"],

                "atomic_actions": atoms,

                "unique_assets": list(unique_assets),

                "D_asset": len(unique_assets),

                "summary_counts": summary_counts
            }