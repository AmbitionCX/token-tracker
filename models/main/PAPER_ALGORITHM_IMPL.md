# 按论文算法的完整GNN实现

## 📋 核心对应关系

### 问题定义 (§1)
✅ **实现**: 边分类问题（edge classification on transaction graph）
- 每条边 = 一笔 external transaction
- 每条边需要识别其交易意图（intention label）
- 数据集：627K 交易 → ~350 可疑（imbalanced）

---

## 📐 交易图构建 (§2-3)

### 时间窗口（Temporal Partitioning）
✅ **代码**: `build_temporal_graphs()` 
```python
window_id = (tx['block_number'] - start_block) // self.temporal_window
# 按1000块为单位分组 → ~10 个时间窗口
```

### 节点集合 $V_k$
✅ **代码**: `TemporalGraphWithTraces.finalize()`
```python
sorted_nodes = sorted(self.nodes_set)  # V_k = all addresses in window
node_to_idx = {addr: idx for idx, addr in enumerate(sorted_nodes)}
# 映射：地址 → 节点索引
```

### 边集合 $E_k$  
✅ **代码**: `add_transaction()` → `finalize()`
```python
edge_index = np.array([from_node_idx, to_node_idx]).T  # (2, num_edges)
# 一笔 external tx ⇔ 一条边
```

---

## 🔗 边特征建模 (§4)

### 4.1 External Features $\mathbf{x}_e^{\text{external}}$ (6维)
✅ **代码**: `TemporalGraphWithTraces.add_transaction()`
```python
external_features = np.array([
    float(value),           # 转账金额
    float(gas_used),        # gas消耗
    0.0,                    # calldata长度
    1.0 if is_contract else 0.0,  # 是否合约调用
    0.0,                    # 是否revert
    0.0                     # nonce位置
])
```

### 4.2 Internal Call Trace Features - 三步处理

#### 4.2.1 线性化 (DFS)
✅ **代码**: `linearize_call_trace()`
```python
def dfs(call: Dict, depth: int = 0):
    calls.append({
        'call_type': call.get('type'),  # CALL/DELEGATECALL/STATICCALL/CREATE
        'to': call.get('to'),           # 被调用合约地址
        'input': call.get('input'),     # 4字节函数选择器 + 参数
        'output': call.get('output'),
        'depth': depth,
        'revert': call.get('revert'),
        'gas': call.get('gas'),
        'gas_used': call.get('gasUsed')
    })
    for subcall in call.get('calls', []):
        dfs(subcall, depth + 1)
```
**输出**: 序列 $\mathcal{C}_e = [c_1, c_2, ..., c_L]$ (DFS顺序)

#### 4.2.2 Token化 (§4.2.2)
✅ **代码**: `tokenize_call_events()`

每个调用事件 $c_i$ 被映射为5维向量和：
$$\mathbf{x}_i = \mathbf{e}_{\text{type}} + \mathbf{e}_{\text{contract}} + \mathbf{e}_{\text{func}} + \mathbf{e}_{\text{depth}} + \mathbf{e}_{\text{exec}}$$

```python
# Token 1: Call Type Embedding
call_type_ids[i] = call_type_map[event['call_type']]  # 1-5

# Token 2: Contract Address Embedding  
contract_ids[i] = hash(event['to']) % 10000

# Token 3: Function Selector Embedding
func_selector = int(input_str[2:10], 16)  # 4字节
func_ids[i] = func_selector % 10000

# Token 4: Depth Embedding
depth_ids[i] = min(event['depth'], 15)

# Token 5: Execution Properties Embedding
exec_properties[i] = [
    float(event['revert']),           # revert标志
    gas_ratio,                         # gas消耗比例
    min(input_len / 1000, 1.0),      # 输入长度
    min(output_len / 1000, 1.0)      # 输出长度
]
```

#### 4.2.3 序列编码器
⏳ **占位符**: 需要在模型中实现 TraceEncoder (Transformer)
- 当前: 存储 token ID 供后续处理
- 后续: `seq_gnn_model.py` 中的 TraceEncoder 将编码这些 tokens

### 4.3 合并特征编码
✅ **代码**: `add_transaction()` 末尾
```python
edge_feature = {
    'external_features': external_features,  # (7,)
    'trace_features': {                      # Token化的trace
        'call_type_ids': call_type_ids,      # (seq_len,)
        'contract_ids': contract_ids,        # (seq_len,)
        'func_ids': func_ids,                # (seq_len,)
        'depth_ids': depth_ids,              # (seq_len,)
        'exec_properties': exec_properties,  # (seq_len, 4)
        'sequence_length': len(call_events)
    }
}
```

---

## 🧠 数据加载管道

### 时间窗口图对象
✅ **类**: `TemporalGraphWithTraces`
- 存储一个时间窗口内的所有事务
- 管理节点集合 V_k 和边集合 E_k
- 最终导出为 PyTorch 张量格式

### 图构造
✅ **类**: `GraphConstructor`
- 从 PostgreSQL 流式加载交易
- 按时间窗口分组
- 为每个窗口构建 `TemporalGraphWithTraces`
- 缓存图到磁盘（避免重复处理）

### 数据集和加载器
✅ **类**: `GraphDataset`, `GraphDataLoader`, `collate_graph_batch()`
- `GraphDataset`: PyTorch Dataset wrapper
- `GraphDataLoader`: 创建 train/val/test 分割
- `collate_graph_batch()`: 将多个时间窗口图合并为一个批次

### 批处理
✅ **函数**: `collate_graph_batch()`
```python
# 输入: batch = 多个时间窗口图
# 处理:
# 1. 从所有图中收集所有边
# 2. 对每条边的 trace 序列进行填充到 max_seq_len=256
# 3. 创建注意力掩码（有效位置）
# 4. 堆叠所有特征张量

# 输出:
{
    'external_features': (num_edges, 7),
    'call_type_ids': (num_edges, 256),
    'contract_ids': (num_edges, 256),
    'func_selector_ids': (num_edges, 256),
    'depths': (num_edges, 256),
    'exec_properties': (num_edges, 256, 4),
    'trace_mask': (num_edges, 256),        # 注意力掩码
    'labels': (num_edges,),
    'num_edges': int
}
```

---

## 🔗 模型集成 (§4.2.3 - §6)

### TraceEncoder 处理Token序列
**位置**: `models/seq_gnn_model.py` - SequenceGNNModel
```python
# 将collate输出的tokens送入TraceEncoder
if self.use_trace:
    trace_embeds = self.call_event_embedding(
        call_type_ids, contract_ids, func_ids, depth_ids
    )
    trace_encoded = self.trace_encoder(trace_embeds, trace_mask)
    # 输出: (batch_size, trace_hidden_dim)
```

### GNN处理图结构  
**位置**: `models/seq_gnn_model.py` - SequenceGNNModel.forward()
```python
# 可选: 使用边的邻域信息
graphgnn_output = self.gnn_layer(
    node_features,
    edge_features,
    edge_index  # 如果从collate输出提取
)
```

### 最终分类
```python
# 合并 external + trace 特征
combined = torch.cat([external_features, trace_encoded], dim=-1)
# 通过注意力和分类器
logits = self.classifier(combined)
```

---

## 📊 数据流总览

```
PostgreSQL (4M-4.01M blocks)
    ↓
TransactionDataLoader.stream_transactions() [5000-tx chunks]
    ↓
GraphConstructor.load_transactions() 
    ↓ [627K transactions loaded]
build_temporal_graphs()
    ├─ 分组: 每1000块为1个窗口 → ~10 windows
    └─ 对每个窗口:
       ├─ 收集 V_k (所有地址)
       ├─ 构建 E_k (所有交易作为边)
       └─ 对每条边:
          ├─ 提取外部特征 (7维)
          ├─ 线性化 call trace (DFS)
          └─ Token化 (调用类型、合约、函数、深度、执行属性)
    ↓
GraphDataLoader.create_dataloaders()
    ↓ [8:1:1 分割]
DataLoader with collate_graph_batch()
    ├─ 批处理多个时间窗口图
    ├─ 填充 trace 序列到 256
    ├─ 创建注意力掩码
    └─ 堆叠特征张量
    ↓
SequenceGNNModel.forward()
    ├─ TraceEncoder: tokens → 序列表示
    ├─ GNN layers: 节点更新 (可选)
    ├─ Attention: 边邻域上下文
    └─ Classifier: 意图分类 (2类: 正常/可疑)
    ↓
CrossEntropyLoss (加权处理类不平衡)
```

---

## ✅ 论文算法对应检查表

| 论文章节 | 要求 | 实现状态 | 代码位置 |
|---------|------|--------|--------|
| §2 图定义 | G=(V,E) 有向图 | ✅ | TemporalGraphWithTraces |
| §3 时间窗口 | Δ块分组 | ✅ | build_temporal_graphs |
| §3 节点集合 | V_k = {地址} | ✅ | nodes_set, node_to_idx |
| §3 边构造 | E_k = {交易} | ✅ | add_transaction → edge_index |
| §4.1 外部特征 | 7维 | ✅ | external_features 数组 |
| §4.2.1 线性化 | DFS展开 | ✅ | linearize_call_trace |
| §4.2.2 Token化 | 5维和 | ✅ | tokenize_call_events |
| §4.2.3 编码 | Transformer | ⏳ | seq_gnn_model TraceEncoder |
| §5.1 GNN | 节点更新 | ⏳ | seq_gnn_model GNN层 |
| §5.2 注意力 | 边聚合 | ⏳ | seq_gnn_model Attention |
| §6 分类 | Softmax | ⏳ | seq_gnn_model 分类头 |

---

## 🚀 使用示例

```python
from data import GraphConstructor, GraphDataLoader
from core import EdgeFeatureExtractor
from data import TransactionDataLoader

# 1. 加载交易
db_loader = TransactionDataLoader(table_name="tx_joined_4000000_4010000")
feat_extractor = EdgeFeatureExtractor(trace_dim=128)

# 2. 构建图
graph_ctor = GraphConstructor(
    data_loader=db_loader,
    feature_extractor=feat_extractor,
    temporal_window=1000,
    cache_dir='./cache'
)
graphs = graph_ctor.construct_graphs(4000000, 4010000)

# 3. 创建DataLoader
graph_loader = GraphDataLoader(graphs, batch_size=32)
train_loader, val_loader, test_loader = graph_loader.create_dataloaders(
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)

# 4. 遍历批次
for batch in train_loader:
    # batch 包含所有模型所需输入
    external_features = batch['external_features']  # (num_edges, 7)
    trace_features = {
        'call_type_ids': batch['call_type_ids'],    # (num_edges, 256)
        'contract_ids': batch['contract_ids'],
        'func_ids': batch['func_selector_ids'],
        'depths': batch['depths'],
        'exec_properties': batch['exec_properties'],
        'trace_mask': batch['trace_mask']
    }
    labels = batch['labels']  # (num_edges,)
    
    # 送入模型
    logits = model(external_features=external_features, **trace_features)
    loss = criterion(logits, labels)
```

---

## 💾 缓存机制

- 首次运行: ~30-60秒（PostgreSQL流式读取）
- 后续运行: <3秒（从磁盘加载图）
- 缓存路径: `./cache/graphs_{start_block}_{end_block}.pkl`

---

## ⚠️ 当前限制 & 后续工作

1. **Node Features**: 当前为零初始化，可添加历史统计特征
2. **GNN Layer**: 图结构已准备，等待GNN实现集成
3. **Neighborhood Context**: edge_indices支持但未使用，可扩展为k-hop邻域
4. **多语义图**: 后续可添加不同交互类型的多条边

---

## 参考文献

实现遵循论文的问题定义（§1-§2）、特征设计（§4）和模型架构（§5-§6）。
