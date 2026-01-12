
# Representation Module

本模块实现论文中 **Section 4.2 行为级表示学习（Behavior-level Representation Learning）** 的前两步：
- **4.2.2 行为级 Token 化与特征编码**
- **4.2.3 序列编码器（TraceEncoder，尚未接入模型训练）**

当前代码已完成从 **原始 EVM 调用轨迹** 到 **可用于序列建模的行为 token 序列** 的完整数据准备流程。


## 1. 输入数据格式

输入为单笔交易的线性化 call trace，示例如下（`first_tx_calls.json`）：

```json
{
  "blockNumber": ...,
  "txHash": "...",
  "L": 48,
  "C_e": [
    {
      "id": 0,
      "parentId": -1,
      "depth": 0,
      "callType": "CALL",
      "callee": "0x...",
      "selector": "0x2b604585",
      "reverted": false,
      "inputLength": 682,
      "outputLength": 0,
      "gasUsed": 695787
    },
    ...
  ]
}
```

其中 `C_e = [c_1, ..., c_L]` 为按 **DFS 执行顺序** 线性化后的调用事件序列。


## 2. 4.2.2 行为级 Token 化（Tokenization）

### 2.1 设计思想

每个调用事件 `c_i` 被视为一个 **行为 token**，由多个语义子特征组成，而非单一离散符号：

* 调用类型（CALL / STATICCALL / …）
* 被调用合约地址
* 函数选择器（4-byte selector）
* 调用深度（call depth）
* 执行属性（是否 revert、输入输出规模、gas 消耗）

论文中对应公式：
$$\mathbf{x}_i
= \mathbf{e}_{type}+ \mathbf{e}_{contract}+ \mathbf{e}_{func}+ \mathbf{e}_{depth}+ \mathbf{e}_{exec}$$

当前阶段实现的是 **离散化 + index 化**（embedding 在后续 PyTorch 中完成）。


### 2.2 代码结构

```
representation/
├─ tokenization/
│   ├─ buildTokens.js        # 主入口：C_e → token sequence
│   ├─ encodeCallEvent.js    # 单个调用事件编码
│   ├─ discretize.js         # 连续特征离散化（gas / length）
│   └─ vocab.js              # 动态构建离散词表
│
└─ test.js                   # tokenization 测试与导出
```




### 2.3 buildTokenSequence 接口

```
buildTokenSequence(C_e) → {
  tokens: Array<Object>,
  vocabs: {
    typeVocab,
    contractVocab,
    funcVocab
  }
}
```

* `C_e`：调用事件数组
* `tokens[i]`：第 i 个行为 token 的离散特征表示
* `vocabs`：当前交易构建的离散词表（后续可升级为全局词表）


## 3. test.js：完整 Tokenization 流程

`test.js` 完成以下步骤：

1. 读取 `first_tx_calls.json`
2. 调用 `buildTokenSequence(tx.C_e)`
3. 组织实验级输出结构
4. 保存为 `token_sequence.json`

输出格式示例：

```
{
  "blockNumber": ...,
  "txHash": "...",
  "L": 48,
  "tokenSequence": [ ... ],
  "vocabSize": {
    "type": 4,
    "contract": 9,
    "function": 7
  }
}
```



## 4. 当前完成状态

* [x] 调用树 DFS 线性化（trace_process）
* [x] 行为级 token 定义与离散编码
* [x] 序列化输出（JSON）
* [ ] 全数据集 vocab 构建
* [ ] PyTorch Dataset
* [ ] Transformer TraceEncoder
* [ ] 下游任务（意图识别 / 分类）

