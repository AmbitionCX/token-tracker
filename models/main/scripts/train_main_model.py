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
from training import Trainer, compute_metrics, TrainingVisualizer
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
        test_ratio=1.0 - float(data_cfg['train_ratio']) - float(data_cfg['val_ratio'])
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
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(training_cfg['num_epochs']),
        learning_rate=float(training_cfg['learning_rate']),
        weight_decay=float(training_cfg['weight_decay']),
        patience=int(training_cfg['early_stopping']['patience']),
        save_best=training_cfg['checkpoint']['keep_best']
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
            loss_fn=torch.nn.CrossEntropyLoss(),
            metric_fn=lambda preds, labels: compute_metrics(preds, labels)
        )
        
        print("\nTest Results:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
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
            y_scores_final=y_test_scores
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
