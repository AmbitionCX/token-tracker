# Scripts for Training and Experiments

This directory contains executable scripts for training models and running experiments.

## Available Scripts

### 1. `train_main_model.py`
**Train the proposed SequenceGNN model.**

```bash
python scripts/train_main_model.py
```

Features:
- Loads configuration from `config.yaml`
- Streams data from PostgreSQL
- Builds temporal graphs with 1000-block windows
- Trains SequenceGNN with Transformer trace encoding
- Evaluates on validation and test sets
- Saves best checkpoint

### 2. `run_gnn_baselines.py`
**Train three GNN baselines (external-only, + mean trace, + Transformer) and print test metrics.**

```bash
cd models/main
python scripts/run_gnn_baselines.py
```

Outputs:
- Checkpoints under `checkpoints/gnn_baselines/<experiment>/`
- `checkpoints/gnn_baselines/gnn_baselines_metrics.csv` (Accuracy, PR-AUC, Precision, Recall, F1, AUC-ROC at threshold 0.5)

Source models:
- `baselines/gnn_model.py` — GNN (External only)
- `baselines/gnn_mean_trace_model.py` — GNN + Mean Trace (pooling encoder)
- `baselines/gnn_transformer_model.py` — GNN + Transformer

### 3. `train_baselines.py` (TO IMPLEMENT)
**Train all baseline models.**

```bash
python scripts/train_baselines.py
```

Baselines:
- Logistic Regression (external features only)
- Random Forest
- XGBoost
- Pure GCN (graph-only)
- Pure GraphSAGE (graph-only)
- LSTM (sequence-only)
- Transformer (sequence-only)

### 4. `run_ablations.py` (TO IMPLEMENT)
**Run all 6 ablation experiments.**

```bash
python scripts/run_ablations.py
```

Experiments:
1. **Complete Model**: Full SequenceGNN (baseline)
2. **No Trace**: Remove trace features
3. **No GNN**: Remove graph component
4. **No Attention**: Remove attention aggregation
5. **LSTM Encoder**: LSTM instead of Transformer
6. **Pooling Encoder**: Pooling instead of Transformer

### 5. `generate_report.py` (TO IMPLEMENT)
**Generate comprehensive results report.**

```bash
python scripts/generate_report.py
```

Outputs:
- Metrics comparison table (CSV)
- ROC curves (PNG)
- Confusion matrices
- Ablation study summary
- Performance analysis

## Configuration

All scripts read from `config.yaml` in the parent directory.

Key settings:
```yaml
data:
  table_name: "tx_joined_4000000_4010000"
  start_block: 4000000
  end_block: 4010000
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  positive_weight: 10.0
```

## Example Usage

### Train main model
```bash
cd models/main
python scripts/train_main_model.py
```

### Check results
```bash
ls checkpoints/  # View saved models
tail -f logs/train.log  # Monitor training
tensorboard --logdir logs  # View metrics
```

### Inspect specific model
```python
import torch
checkpoint = torch.load('checkpoints/best_model.pt')
print(checkpoint.keys())  # epoch, global_step, model_state
```

## Output Directories

```
models/main/
├── checkpoints/          # Saved model weights
│   ├── best_model.pt
│   └── final_model.pt
├── logs/                 # Training logs
│   ├── train.log
│   └── runs/             # TensorBoard event files
└── results/              # Experiment results (auto-created)
    ├── metrics.csv
    ├── confusion_matrices.png
    └── ablation_summary.json
```

## Tips

1. **Monitor Training**: Use TensorBoard
   ```bash
   tensorboard --logdir logs/runs/
   ```

2. **Debug Data Loading**: Check database connection
   ```python
   from data import TransactionDataLoader
   loader = TransactionDataLoader()
   count = loader.get_transaction_count(4000000, 4010000)
   print(f"Transactions: {count}")
   ```

3. **Check Model Size**: Estimate GPU memory
   ```python
   from models import SequenceGNNModel
   model = SequenceGNNModel()
   params = sum(p.numel() for p in model.parameters())
   print(f"Parameters: {params / 1e6:.1f}M")  # Millions
   ```

4. **Resume Training**: Load checkpoint
   ```python
   # In trainer.py
   trainer.load_checkpoint('best_model.pt')
   # Continue training with trainer.fit(...)
   ```

## Troubleshooting

**GPU Out of Memory**
- Reduce `batch_size` in config.yaml
- Reduce `trace_hidden_dim` or `gnn_hidden_dim`
- Reduce window size (`data.window_size`)

**Slow Data Loading**
- Enable caching: `data.cache_enabled: true`
- Increase `chunk_size` for streaming
- Use SSD for cache directory

**Poor Model Performance**
- Check label distribution (is_suspicious imbalance)
- Increase `positive_weight` for suspicious class
- Tune learning rate or add learning rate scheduler
- Check if trace features are being extracted correctly

## Next Steps

After implementing scripts:
1. Run baseline models and compare
2. Run ablation experiments
3. Generate comparison report
4. Analyze which components contribute most to performance
5. Fine-tune hyperparameters based on validation results
