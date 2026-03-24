"""
Visualization script for training results.

Usage:
    python scripts/visualize_results.py [checkpoint_dir] [output_dir]

Example:
    python scripts/visualize_results.py ./checkpoints ./results
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

from training import TrainingVisualizer
from training.metrics import compute_metrics


def load_training_history(checkpoint_dir: str) -> dict:
    """
    Load training history from checkpoint directory.
    
    Looks for:
    - train_loss.json
    - val_loss.json
    - val_metrics.json
    """
    checkpoint_path = Path(checkpoint_dir)
    
    history = {}
    
    # Try to load loss history
    if (checkpoint_path / 'history.json').exists():
        with open(checkpoint_path / 'history.json', 'r') as f:
            history = json.load(f)
    else:
        print(f"Warning: history.json not found in {checkpoint_dir}")
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    return history


def visualize_from_checkpoint(
    checkpoint_dir: str = './checkpoints',
    output_dir: str = './results',
    history_file: str = 'history.json',
    metrics_file: str = 'test_metrics.json'
) -> None:
    """
    Generate visualizations from saved checkpoint and metrics.
    
    Args:
        checkpoint_dir: Directory containing saved model and history
        output_dir: Directory to save visualizations
        history_file: Name of training history file
        metrics_file: Name of test metrics file
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading training history from {checkpoint_path / history_file}...")
    
    # Load history
    if (checkpoint_path / history_file).exists():
        with open(checkpoint_path / history_file, 'r') as f:
            history = json.load(f)
    else:
        print(f"Error: {history_file} not found in {checkpoint_dir}")
        return
    
    # Load test metrics if available
    test_data = None
    if (checkpoint_path / metrics_file).exists():
        with open(checkpoint_path / metrics_file, 'r') as f:
            test_data = json.load(f)
        print(f"Loading test metrics from {checkpoint_path / metrics_file}...")
    
    # Create visualizer
    visualizer = TrainingVisualizer(output_dir=str(output_path))
    
    # Plot loss curves
    if 'train_loss' in history and 'val_loss' in history:
        print("Generating loss curves...")
        visualizer.plot_loss_curves(history)
    
    # Plot metrics curves
    if 'val_metrics' in history:
        print("Generating metrics curves...")
        visualizer.plot_metrics_curves(history)
    
    # Plot test results if available
    if test_data:
        y_true = np.array(test_data.get('y_true', []))
        y_pred = np.array(test_data.get('y_pred', []))
        y_scores = np.array(test_data.get('y_scores', []))
        
        if len(y_true) > 0:
            print("Generating confusion matrix...")
            visualizer.plot_confusion_matrix(y_true, y_pred)
            
            if len(y_scores) > 0:
                print("Generating ROC curve...")
                visualizer.plot_roc_curve(y_true, y_scores)
                print("Generating PR curve...")
                visualizer.plot_precision_recall_curve(y_true, y_scores)
            
            print("Generating class distribution...")
            visualizer.plot_class_distribution(y_true)
            
            print("Generating summary report...")
            visualizer.generate_summary_report(history, y_true, y_pred, y_scores)
    
    print(f"\n✓ All visualizations saved to {output_path}")


if __name__ == '__main__':
    checkpoint_dir = './checkpoints' if len(sys.argv) < 2 else sys.argv[1]
    output_dir = './results' if len(sys.argv) < 3 else sys.argv[2]
    
    print("=" * 60)
    print("Training Results Visualization")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    visualize_from_checkpoint(checkpoint_dir, output_dir)
