from openai import OpenAI
import pandas as pd
import json

# ========== 初始化 API ==========
client = OpenAI(
    api_key="sk-vviolebrfvryzgkwpaahipiwgoijlcnhwpelwtjnfzvysvhy",  # 你的 key
    base_url="https://api.siliconflow.cn/v1",
)

# ========== 加载数据 ==========
df = pd.read_csv("../joined_results.csv")
df = df.head(100)   # ⭐ 只取前 10 条

system_prompt = """
你是链上交易意图分析引擎。你将根据输入交易信息（transaction_hash, trace, function_sequence, id, block_number, transaction_index, from_address, to_address, contract_address, gas_used, cumulative_gas_used, gas_price, logs, timestamp）对每笔交易进行意图识别，并输出意图类型与理由说明。

可使用的线索包括但不限于：函数名、trace调用链、内部调用、ERC20事件、Approval、LP mint/burn、借贷资产变化、奖励领取、桥接流程、DEX路由行为、MEV特征、是否NFT合约、是否流动性池、是否staking池等。

你必须从下列标签中选择最符合的一项作为最终意图，注意你只能从下面选择，禁止自己编造下面不存在的类型：

资金转移类：

* transfer（普通代币转账）
* batch_transfer（批量分发或批转）
* self_transfer（地址间自转存储或调仓）

交易类（Swap/Trade）：

* swap（Token兑换，调用DEX Router）
* limit_order_execution（限价订单成交、RFQ执行）
* aggregator_swap（多路由swap，如1inch/CowSwap）

流动性相关：

* add_liquidity（向LP池提供资金获取LP token）
* remove_liquidity（撤池并销毁LP token取回资产）
* zap_in（单token转多token后入池）
* zap_out（退出LP并拆回token）

授权与权限：

* approve（授权代币使用额度）
* revoke_approval（撤销授权）
* set_allowance(increase/decrease)（调整授权额度）

质押收益/挖矿：

* stake（质押资产进入staking池）
* unstake（从池子赎回本金）
* claim（领取收益奖励）
* compound（复投策略，将收益重新投入）

借贷/清算类：

* lend/deposit（存入借贷协议）
* borrow（借出资产）
* repay（归还借款）
* liquidation（清算他人仓位行为）

DeFi协议交互：

* vault_deposit（存入vault策略池）
* vault_withdraw（从vault取回资产）
* yield_farm_interaction（农场交互但意图未细分）
* bribe/governance_vote（治理投票相关）
* reward_distribution（收益派发）

NFT & GameFi：

* nft_mint（铸造NFT）
* nft_trade（购买或出售NFT）
* nft_airdrop_receive（空投NFT）
* game_asset_transfer（链游资产转账）

跨链/桥接：

* bridge_lock（资产锁入跨链桥）
* bridge_mint（跨链到另一链后mint资产）
* bridge_burn（burn资产并跨链赎回）
* bridge_redeem（赎回跨链资产）

MEV/套利/机器人行为：

* arbitrage（多路径价格套利）
* sandwich_attack（三明治攻击结构）
* liquidation_bot（清算机器人）
* mev_bundle_execution（MEV打包行为）

内部系统行为：

* token_mint（增发代币）
* token_burn（销毁代币）
* reflection_distribution（持币返利）
* fee_charging（手续费扣取）

高风险/攻击/异常：

* rugpull(liquidity_withdrawal)（流动性抽离）
* honeypot（买入可卖出失败行为）
* phishing_approval（恶意合约授权）
* suspicious_mass_outflow（异常大量资金迁移）
* exploit_attack（合约攻击利用）
* unknown_high_risk（高度可疑但意图不明）

无法判断：

* unknown（信息不足或不符合任何模式）

意图判定逻辑要求简明可靠：

* 根据 function_sequence、trace 调用结构定位行为类型
* 根据 logs 中 Transfer/Approval/LP mint/LP burn 判断资金流向
* 根据池子合约、借贷协议、NFT合约、跨链桥等模式进行分类
* 优先命中意图特征明显的标签，否则使用 unknown

输出必须为 JSON，且仅包含 intent 和 reason，不得包含分析文本或其他内容，格式如下：
{"intent": "...", "reason": "..."}
"""

def label_tx(row):
    tx = {
        "transaction_hash": row.get("transaction_hash"),
        "function_sequence": row.get("function_sequence"),
        "trace": row.get("trace")
    }

    prompt = f"请分析以下交易并打标签：\n{json.dumps(tx, ensure_ascii=False)}"

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        print("API 调用失败：", e)
        return "unknown", str(e)

    content = response.choices[0].message.content.strip()

    # 尝试解析 JSON
    try:
        j = json.loads(content)
        return j.get("intent", "unknown"), j.get("reason", "")
    except Exception:
        # 如果模型没返回正常 JSON，就原样打印
        return "unknown", content


# ========== 逐条调用，并打印结果 ==========
results = []
for i, row in df.iterrows():
    print(f"\n===== 处理第 {i+1} 条交易 =====")
    intent, reason = label_tx(row)
    print("transaction_hash:", row["transaction_hash"])
    print("intent:", intent)
    print("reason:", reason)
    results.append((intent, reason))


# 如果你想之后保存，只要取消注释即可
# df["intent"] = [r[0] for r in results]
# df["reason"] = [r[1] for r in results]
# df.to_csv("labeled_intents_debug.csv", index=False, encoding="utf-8")
# print("\n调试标注完成，已写入 labeled_intents_debug.csv")
