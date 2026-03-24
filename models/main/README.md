# GNN-Based Suspicious Transaction Detection

## Project Overview

A comprehensive Graph Neural Network system for detecting suspicious transactions on Ethereum using:
1. **Sequence modeling** of internal call traces (Transformer/LSTM/Pooling)
2. **Graph Neural Networks** for address node representation (GAT/GCN/GraphSAGE)
3. **Edge-aware attention** for local structural pattern recognition
4. **Temporal windowing** for localized temporal graphs

## Architecture

```
Raw Transaction
     ↓
[External Features] ← 4 dimensions
     ↓
[Internal Call Trace] → Linearize (DFS) → Tokenize → Encode (Transformer/LSTM/Pooling)
     ↓
[Combined Edge Features] 
     ↓
[GNN Module] → Node Representations
     ↓
[Edge Attention Aggregation] → Incorporate neighborhood context
     ↓
[Classifier] → Binary Classification (Suspicious/Normal)  /Multi Classification
```

## Project Structure

```
models/main/
├── config.yaml                        # Global configuration
├── requirements.txt                   # Dependencies
├── README.md                          # This file
│
├── core/                              # Core reusable components
│   ├── trace_encoder.py              # Transformer/LSTM/Pooling for call traces
│   ├── edge_feature_extractor.py     # External + trace feature extraction
│   ├── temporal_graph_builder.py     # Temporal window graph construction
│   ├── attention_aggregator.py       # Edge attention aggregation
│   └── __init__.py
│
├── models/                            # Model architectures
│   ├── gnn_base.py                   # GNN modules (GAT/GCN)
│   ├── seq_gnn_model.py              # Main SequenceGNN model + ablations
│   └── __init__.py
│
├── baselines/
│   ├── __init__.py
│   ├── xgboost_model.py              # XGBoost (External only)
│   ├── mlp_model.py                  # MLP (+ Mean Trace)
│   ├── lstm_model.py                 # LSTM (Trace only)
│   ├── transformer_model.py          # Transformer (Trace only)
│   ├── gnn_model.py                  # GNN (External only)
│   ├── gnn_mean_trace_model.py       # GNN + Mean Trace
│   ├── gnn_transformer_model.py      # GNN + Transformer
│
├── data/                              # Data loading & preprocessing
│   ├── data_loader.py                # PostgreSQL streaming + caching
│   ├── graph_constructor.py          # Build temporal graphs from batches
│   └── __init__.py
│
├── training/                          # Training framework
│   ├── trainer.py                    # Unified trainer (PyTorch + sklearn)
│   ├── metrics.py                    # Evaluation metrics
│   ├── callbacks.py                  # Early stopping, logging
│   └── __init__.py
│
├── experiments/                       # Ablation experiments
│   ├── ablation_configs.py          # 6 ablation configurations
│   ├── exp_manager.py               # Run experiments sequentially
│   ├── results_analyzer.py          # Aggregate & visualize results
│   └── __init__.py
│
└── scripts/                           # Execution scripts
    ├── train_baselines.py           # Train all baselines
    ├── train_main_model.py          # Train main SequenceGNN model
    ├── run_ablations.py             # Run all 6 ablations
    └── generate_report.py           # Generate results report
```

## Key Components

### 1. Core Modules (Phase 1)

#### `trace_encoder.py`
- **TransformerTraceEncoder**: Self-attention over call events
- **LSTMTraceEncoder**: Recurrent sequence modeling
- **PoolingTraceEncoder**: Simple mean/max pooling baseline
- **CallEventEmbedding**: Multi-semantic dimension embedding

#### `edge_feature_extractor.py`
- **ExternalTransactionFeatures**: 7-dimensional external features
  - `value`, `gas_used`, `gas_price`, `calldata_length`,
  - `is_contract_call`, `is_revert`, `nonce_position`
- **InternalCallTraceFeatures**: DFS linearization + tokenization
  - Call type, contract address, function selector, depth, execution properties
- **EdgeFeatureExtractor**: Combined 135-dimensional edge features

#### `temporal_graph_builder.py`
- **TemporalGraphBuilder**: Partition into Δ=1000 block windows
- **TemporalWindow**: Single window with nodes/edges
- **MultiWindowGraphDataset**: PyTorch Dataset wrapper

#### `attention_aggregator.py`
- **EdgeAttentionAggregator**: Multi-head attention over neighborhoods
- **EdgeRepresentationAggregator**: Complete aggregation pipeline
- **NeighborhoodContextExtractor**: Build k-hop neighborhoods

### 2. Model Architectures (Phase 2)

#### `gnn_base.py`
- **GATConvLayer**: Graph Attention Network layer
- **GCNConvLayer**: Graph Convolutional Network layer
- **GNNBase**: Unified interface supporting GAT/GCN/GraphSAGE

#### `seq_gnn_model.py`
- **SequenceGNNModel**: Main model combining all components
- **Ablation Variants**:
  - `NoTraceSequenceGNN`: Remove trace features
  - `NoGNNSequenceModel`: Remove graph component
  - `NoAttentionSequenceGNN`: No attention aggregation
  - `LSTMSequenceGNN`: LSTM instead of Transformer
  - `PoolingSequenceGNN`: Pooling instead of Transformer

#### Baseline Models (aligned with Table 1)
- `xgboost_model.py`: XGBoost using external transaction features only
- `mlp_model.py`: MLP with external + mean-pooled trace features
- `lstm_model.py`: LSTM encoder over call trace sequences
- `transformer_model.py`: Transformer encoder over call trace sequences
- `gnn_model.py`: GNN using external features only
- `gnn_mean_trace_model.py`: GNN with mean-pooled trace features
- `gnn_transformer_model.py`: GNN with Transformer-based trace encoding

### 3. Data Processing (Phase 4)

#### `data_loader.py`
- **TransactionDataLoader**: PostgreSQL streaming with JSON parsing
- **CachedDataLoader**: Disk-based caching for efficiency
- **stream_prepared_transactions**: Feature extraction on-the-fly

### 4. Training Framework (Phase 4)

#### `trainer.py`
- unified `Trainer` class supporting PyTorch models
- Methods: `train_epoch()`, `validate()`, `fit()`
- Features: checkpointing, early stopping, TensorBoard logging

#### `metrics.py`
- **compute_metrics()**: Precision, recall, F1, AUC, confusion matrix
- Handles imbalanced classification with threshold tuning

### 5. Ablation Experiments (Phase 5)

#### `ablation_configs.py`
Predefined experiments:
1. **Complete Model**: Full SequenceGNN
2. **No Trace**: GNN+Attention only
3. **No GNN**: Trace+Attention only  
4. **No Attention**: Trace+GNN only
5. **LSTM Encoder**: Replace Transformer with LSTM
6. **Pooling Encoder**: Replace Transformer with Pooling

## Data Schema

**Input Table**: `tx_joined_4000000_4010000`

| Column | Type | Description |
|--------|------|-------------|
| block_number | INT | Block height |
| transaction_hash | VARCHAR | Transaction hash |
| from_address | VARCHAR | Sender address |
| to_address | VARCHAR | Recipient/contract address |
| gas_used | BIGINT | Gas consumed |
| gas_price | BIGINT | Gas price in wei |
| value | NUMERIC | ETH value transferred |
| logs | JSON | ERC20 transfer logs |
| trace_data | JSON | EVM call trace (nested tree) |
| timestamp | TIMESTAMP | Block timestamp |
| is_suspicious | INT | Label (0=normal, 1=suspicious) |

## Configuration

Edit `config.yaml`:

```yaml
data:
  table_name: "tx_joined_4000000_4010000"
  start_block: 4000000
  end_block: 4010000
  window_size: 1000  # Δ blocks per window

model:
  trace_encoder:
    type: "transformer"  # Options: transformer, lstm, pooling
    hidden_dim: 128
  
  gnn:
    type: "gat"  # Options: gat, gcn, graphsage
    hidden_dim: 128

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  positive_weight: 10.0  # For imbalanced data
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```python
from data import TransactionDataLoader
loader = TransactionDataLoader(table_name="tx_joined_4000000_4010000")
for chunk in loader.stream_transactions(4000000, 4010000, chunk_size=5000):
    print(f"Loaded {len(chunk)} transactions")
```

### 3. Train Main Model
```python
from models import SequenceGNNModel
from training import Trainer

model = SequenceGNNModel()
trainer = Trainer(model)
# trainer.fit(train_loader, val_loader, num_epochs=50)
```

### 4. Run Ablations
```python
from experiments import ABLATION_CONFIGS

for exp_id, config in ABLATION_CONFIGS.items():
    print(f"Running {config.name}...")
    # Run training with config
```

## Performance Metrics

**Binary Classification Metrics**:
- Accuracy
- Precision (FP avoidance)
- Recall (TP capture)
- F1 Score
- ROC-AUC (probability calibration)
- Confusion Matrix (TP/FP/TN/FN)

**Hardware Requirements**:
- Minimum: 4GB GPU (RTX 2060)
- Recommended: 8GB+ GPU (RTX 3060+)
- Training time: ~2-3 hours per model on full dataset

## Ablation Study Design

| Experiment | Configuration | Purpose |
|-----------|---|---------|
| **Complete** | ✅Trace ✅GNN ✅Attn | Baseline (full) |
| **Exp-A** | ✗Trace ✅GNN ✅Attn | Trace importance |
| **Exp-B** | ✅Trace ✗GNN ✅Attn | Graph importance |
| **Exp-C** | ✅Trace ✅GNN ✗Attn | Attention importance |
| **Exp-D-1** | ✅Trace(LSTM) ✅GNN ✅Attn | LSTM vs Transformer |
| **Exp-D-2** | ✅Trace(Pool) ✅GNN ✅Attn | Pooling vs Transformer |

## Expected Results

On test set (Δ=1000 block windows):

| Model | Precision | Recall | F1 | AUC |
|-------|-----------|--------|----|----|
| Logistic Regression (baseline) | 0.65 | 0.45 | 0.53 | 0.72 |
| SequenceGNN (proposed) | **0.78** | **0.68** | **0.73** | **0.84** |
| No-Trace variant | 0.72 | 0.60 | 0.66 | 0.79 |
| No-GNN variant | 0.68 | 0.62 | 0.65 | 0.77 |
| No-Attention variant | 0.74 | 0.65 | 0.69 | 0.81 |

*Note: Actual results depend on data distribution and hyperparameters*

## Future Extensions

1. **Multi-class Classification**: Distinguish attack types (arbitrage, sandwich, etc.)
2. **Temporal Dynamics**: Sequence modeling across time windows
3. **Cross-chain**: Support Solana, Polygon, etc.
4. **Explainability**: Attention visualization for suspicious patterns
5. **Serving**: ONNX export + REST API for real-time detection

## References

- **Paper Section 4.2**: Internal call trace modeling
- **Paper Section 5.1-5.2**: GNN + attention architecture
- **Paper Section 6**: Transaction intention classification

## Contact

For questions or issues, open an issue on the project repository.
