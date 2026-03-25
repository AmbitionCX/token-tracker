"""
Train the main SequenceGNN model.

Example script showing how to:
1. Load data from PostgreSQL
2. Extract features
3. Build temporal graphs
4. Train the model
5. Evaluate on test set
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

from core import (
    TemporalGraphBuilder,
    EdgeFeatureExtractor,
    MultiWindowGraphDataset
)
from models import SequenceGNNModel
from data import (
    TransactionDataLoader,
    GraphConstructor,
    GraphDataLoader
)
from training import (
    Trainer,
    compute_metrics,
    compute_binary_test_metrics,
    TrainingVisualizer,
    build_train_loss,
)
import yaml


def load_config(config_path: str = '../config.yaml') -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple:
    """
    Prepare data for training.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load configuration
    data_cfg = config['data']
    
    # Create data loader
    db_loader = TransactionDataLoader(
        table_name=data_cfg['table_name']
    )
    
    # Get transaction count for splitting
    total_tx = db_loader.get_transaction_count(
        int(data_cfg['start_block']),
        int(data_cfg['end_block'])
    )
    
    print(f"Total transactions: {total_tx}")
    print(f"Label distribution: {db_loader.get_label_distribution(int(data_cfg['start_block']), int(data_cfg['end_block']))}")
    
    # Create feature extractor
    feature_extractor = EdgeFeatureExtractor(
    trace_embedding_dim=128,
    max_trace_length=256,
    use_trace=True,
    use_external=True
    )
    
    # Build temporal graphs
    graph_constructor = GraphConstructor(
        data_loader=db_loader,
        feature_extractor=feature_extractor,
        temporal_window=int(data_cfg.get('temporal_window', 1000)),
        cache_dir=data_cfg.get('cache_dir', './cache')
    )
    
    graphs = graph_constructor.construct_graphs(
        start_block=int(data_cfg['start_block']),
        end_block=int(data_cfg['end_block']),
        use_cache=True,
        force_rebuild=False
    )
    
    print(f"Constructed {len(graphs)} temporal graphs")
    
    # Create DataLoaders from graphs
    graph_loader = GraphDataLoader(
        graphs=graphs,
        batch_size=int(data_cfg.get('batch_size', 32)),
        shuffle=True,
        num_workers=int(data_cfg.get('num_workers', 0))
    )
    
    train_loader, val_loader, test_loader = graph_loader.create_dataloaders(
        train_ratio=float(data_cfg['train_ratio']),
        val_ratio=float(data_cfg['val_ratio']),
        test_ratio=1.0 - float(data_cfg['train_ratio']) - float(data_cfg['val_ratio']),
        train_edge_balance=bool(data_cfg.get('train_edge_balance', False)),
    )
    
    print(f"Created DataLoaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def train_model(config: dict) -> None:
    """
    Main training loop.
    """
    # Load configuration
    model_cfg = config['model']
    training_cfg = config['training']
    
    # Auto-detect device (CUDA if available, otherwise CPU)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("⚠️  WARNING: CUDA not available. Using CPU (may be slow).")
        device = "cpu"
    
    # Create model
    model = SequenceGNNModel(
        # Trace encoding
        trace_encoder_type=model_cfg['trace_encoder']['type'],
        trace_hidden_dim=int(model_cfg['trace_encoder']['hidden_dim']),
        trace_num_layers=int(model_cfg['trace_encoder']['num_layers']),
        trace_num_heads=int(model_cfg['trace_encoder']['num_heads']),
        
        # GNN
        gnn_type=model_cfg['gnn']['type'],
        gnn_hidden_dim=int(model_cfg['gnn']['hidden_dim']),
        gnn_num_layers=int(model_cfg['gnn']['num_layers']),
        gnn_num_heads=int(model_cfg['gnn']['num_heads']),
        
        # Attention
        use_attention=True,
        attn_num_heads=int(model_cfg['attention']['num_heads']),
        
        # Options
        dropout=0.1,
        use_trace=True,
        use_gnn=True
    )
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss: use class weights so rare positives are not ignored (see config training.loss_fn / positive_weight)
    loss_fn = build_train_loss(
        loss_name=str(training_cfg.get("loss_fn", "weighted_crossentropy")),
        device=device,
        positive_weight=float(training_cfg.get("positive_weight", 10.0)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        focal_gamma=float(training_cfg.get("focal_gamma", 2.0)),
    )
    print(
        f"Loss: {training_cfg.get('loss_fn', 'weighted_crossentropy')} "
        f"(positive_weight={training_cfg.get('positive_weight', 10.0)}, "
        f"focal_gamma={training_cfg.get('focal_gamma', 2.0)})"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=training_cfg['checkpoint']['save_dir'],
        use_tensorboard=True
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Train
    print("Starting training...")
    metric_for_best = str(training_cfg.get("early_stopping_metric", "macro_f1"))
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(training_cfg['num_epochs']),
        learning_rate=float(training_cfg['learning_rate']),
        weight_decay=float(training_cfg['weight_decay']),
        loss_fn=loss_fn,
        metric_fn=lambda preds, labels: compute_metrics(preds, labels),
        patience=int(training_cfg['early_stopping']['patience']),
        save_best=training_cfg['checkpoint']['keep_best'],
        metric_for_best=metric_for_best,
    )
    
    # Save final model
    trainer.save_checkpoint('final_model.pt')
    print(f"Training completed. Best model saved to {trainer.checkpoint_dir}/best_model.pt")
    
    # Evaluate on test set
    if test_loader is not None:
        print("\nEvaluating on test set...")
        # Load best model
        trainer.load_checkpoint('best_model.pt')
        
        # Evaluate
        test_metrics = trainer.validate(
            test_loader,
            loss_fn=loss_fn,
            metric_fn=lambda preds, labels: compute_metrics(preds, labels)
        )
        print(f"\nTest weighted loss: {test_metrics['loss']:.6f}")
        
        # Collect all predictions for visualization
        print("\nCollecting predictions for visualization...")
        all_preds = []
        all_labels = []
        all_scores = []
        
        trainer.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                batch_device = {}
                labels = None
                
                for k, v in batch.items():
                    if k == 'labels':
                        labels = v.to(device) if isinstance(v, torch.Tensor) else v
                    elif k == 'num_edges':
                        # Skip num_edges - not needed by model forward
                        continue
                    elif isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(device)
                    else:
                        batch_device[k] = v
                
                # Forward pass
                try:
                    logits = trainer.model(**batch_device)
                    preds = torch.argmax(logits, dim=-1)
                    scores = torch.softmax(logits, dim=-1)[:, 1]  # Positive class probability
                    
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_scores.append(scores.cpu().numpy())
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    continue
        
        # Concatenate all predictions
        y_test_true = np.concatenate(all_labels)
        y_test_pred = np.concatenate(all_preds)
        y_test_scores = np.concatenate(all_scores)

        full_test = compute_binary_test_metrics(y_test_true, y_test_scores, threshold_fixed=0.5)
        print("\nTest set (scores + thresholds):")
        print(f"  ROC-AUC: {full_test['roc_auc']:.4f}  |  PR-AUC: {full_test['pr_auc']:.4f}")
        print("  At threshold 0.5 (argmax):")
        print(
            f"    Precision={full_test['precision_fixed']:.4f}  "
            f"Recall={full_test['recall_fixed']:.4f}  F1={full_test['f1_fixed']:.4f}"
        )
        print(
            f"  At best F1 on PR curve (threshold={full_test['best_threshold_pr_f1']:.6f}):"
        )
        print(
            f"    Precision={full_test['precision_at_best_threshold']:.4f}  "
            f"Recall={full_test['recall_at_best_threshold']:.4f}  "
            f"F1={full_test['f1_at_best_threshold']:.4f}"
        )

        report_dir = Path(training_cfg['checkpoint']['save_dir'])
        def _jsonable_scalar(v):
            if isinstance(v, (float, np.floating)):
                x = float(v)
                return None if np.isnan(x) or np.isinf(x) else x
            if isinstance(v, (int, np.integer)):
                return int(v)
            return v

        test_metrics_path = report_dir / 'test_metrics.json'
        with open(test_metrics_path, 'w', encoding='utf-8') as jf:
            json.dump(
                {
                    'y_true': y_test_true.tolist(),
                    'y_pred': y_test_pred.tolist(),
                    'y_scores': y_test_scores.tolist(),
                    'metrics': {k: _jsonable_scalar(v) for k, v in full_test.items()},
                },
                jf,
                indent=2,
            )
        print(f"\nSaved test predictions + metrics to {test_metrics_path}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualizer = TrainingVisualizer(output_dir=training_cfg['checkpoint']['save_dir'])
        
        plots = visualizer.plot_all_results(
            history=history,
            y_true_final=y_test_true,
            y_pred_final=y_test_pred,
            y_scores_final=y_test_scores
        )
        
        # Generate summary report
        report_path = visualizer.generate_summary_report(
            history=history,
            y_true_final=y_test_true,
            y_pred_final=y_test_pred,
            y_scores_final=y_test_scores,
            score_metrics=full_test,
        )
        
        print("\n" + "=" * 60)
        print("VISUALIZATIONS GENERATED")
        print("=" * 60)
        for plot_name, plot_path in plots.items():
            print(f"  ✓ {plot_name}: {plot_path}")
        print(f"  ✓ Summary report: {report_path}")
        print("=" * 60)
    
    trainer.close()


if __name__ == '__main__':
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    config = load_config(config_path)
    
    print("=" * 60)
    print("SequenceGNN Main Model Training")
    print("=" * 60)
    
    # Train
    train_model(config)
