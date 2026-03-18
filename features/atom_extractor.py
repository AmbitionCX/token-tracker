from collections import defaultdict
import json


TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
APPROVAL_SIG = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"

def extract_atomic_actions(tx):

    user_addr = tx["from_address"].lower()

    logs = tx["logs"]
    trace = tx["trace_data"]

    # 防止 pandas 把 jsonb 读成字符串
    if isinstance(logs, str):
        logs = json.loads(logs)

    if isinstance(trace, str):
        trace = json.loads(trace)

    atoms = []
    summary_counts = defaultdict(int)
    unique_assets = set()

    contract_call_count = 0

    # -------------------------
    # 1️⃣ parse execution trace
    # -------------------------

    if trace:

        for call in trace:

            # 防御异常结构
            if not isinstance(call, dict):
                continue

            call_type = call.get("type")
            to_addr = call.get("to")

            if to_addr:
                to_addr = to_addr.lower()

            if call_type in ["CALL", "STATICCALL"]:
                contract_call_count += 1

                atoms.append({
                    "atom": "contract_call",
                    "to": to_addr
                })

                summary_counts["contract_call"] += 1

            elif call_type == "DELEGATECALL":
                atoms.append({
                    "atom": "delegatecall",
                    "to": to_addr
                })

                summary_counts["delegatecall"] += 1

            elif call_type in ["CREATE", "CREATE2"]:
                atoms.append({
                    "atom": "contract_creation",
                    "to": to_addr
                })

                summary_counts["contract_creation"] += 1

    # -------------------------
    # 2️⃣ parse ERC20 logs (Transfer and Approval)
    # -------------------------

    if logs:

        for log in logs:

            if not isinstance(log, dict):
                continue

            topics = log.get("topics", [])
            if len(topics) >= 3:
                # Check for Transfer event
                if topics[0].lower() == TRANSFER_SIG:
                    token = log.get("address", "").lower()
                    from_addr = "0x" + topics[1][-40:]
                    to_addr = "0x" + topics[2][-40:]

                    if from_addr == user_addr:
                        atoms.append({
                            "atom": "token_out",
                            "token": token
                        })

                        summary_counts["token_out"] += 1
                        unique_assets.add(token)

                    if to_addr == user_addr:
                        atoms.append({
                            "atom": "token_in",
                            "token": token
                        })

                        summary_counts["token_in"] += 1
                        unique_assets.add(token)

                # Check for Approval event
                if topics[0].lower() == APPROVAL_SIG:
                    token = log.get("address", "").lower()
                    owner = "0x" + topics[1][-40:]

                    if owner == user_addr:
                        atoms.append({
                            "atom": "token_approval",
                            "token": token
                        })

                        summary_counts["token_approval"] += 1

    # -------------------------
    # 3️⃣ multi-call detection
    # -------------------------

    if contract_call_count >= 2:
        atoms.append({
            "atom": "multi_call",
            "count": contract_call_count
        })

        summary_counts["multi_call"] = 1

    return atoms, dict(summary_counts), unique_assets