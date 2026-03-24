"""
Training results visualization module.

Generates plots for:
- Loss curves (train/val)
- Metric curves (precision, recall, F1, AUC)
- Confusion matrix
- ROC curve
- Class distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve
)
import json


class TrainingVisualizer:
    """Generate visualizations for training results."""
    
    def __init__(self, output_dir: str = './results'):
        """
        Args:
            output_dir: Directory to save visualization figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_loss_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training and Validation Loss",
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot training and validation loss curves.
        
        Args:
            history: Dict with keys 'train_loss', 'val_loss'
            title: Plot title
            save_path: Path to save figure (if None, uses default)
        
        Returns:
            Path where figure was saved
        """
        if save_path is None:
            save_path = self.output_dir / "loss_curves.png"
        else:
            save_path = Path(save_path)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = np.arange(len(history['train_loss']))
        ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Loss curves saved to {save_path}")
        return str(save_path)
    
    def plot_metrics_curves(
        self,
        history: Dict[str, List[Dict[str, float]]],
        metrics: List[str] = ['precision', 'recall', 'f1', 'auc'],
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot metric curves over epochs.
        
        Args:
            history: Dict with key 'val_metrics' containing list of metric dicts
            metrics: List of metric names to plot
            save_path: Path to save figure
        
        Returns:
            Path where figure was saved
        """
        if save_path is None:
            save_path = self.output_dir / "metrics_curves.png"
        else:
            save_path = Path(save_path)
        
        # Extract metrics
        epochs = np.arange(len(history['val_metrics']))
        metric_values = {m: [] for m in metrics}
        
        for epoch_metrics in history['val_metrics']:
            for metric in metrics:
                if metric in epoch_metrics:
                    metric_values[metric].append(epoch_metrics[metric])
                else:
                    metric_values[metric].append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, metric in enumerate(metrics):
            if metric in metric_values:
                ax.plot(
                    epochs,
                    metric_values[metric],
                    'o-',
                    label=metric.upper(),
                    linewidth=2,
                    markersize=4,
                    color=colors[i % len(colors)]
                )
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Validation Metrics Over Epochs', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Metrics curves saved to {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save figure
        
        Returns:
            Path where figure was saved
        """
        if save_path is None:
            save_path = self.output_dir / "confusion_matrix.png"
        else:
            save_path = Path(save_path)
        
        if class_names is None:
            class_names = ['Normal', 'Suspicious']
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add metrics text
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}'
        plt.text(1.5, -0.3, metrics_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {save_path}")
        return str(save_path)
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels (binary)
            y_scores: Predicted scores/probabilities for positive class
            save_path: Path to save figure
        
        Returns:
            Path where figure was saved
        """
        if save_path is None:
            save_path = self.output_dir / "roc_curve.png"
        else:
            save_path = Path(save_path)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curve saved to {save_path}")
        return str(save_path)
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels (binary)
            y_scores: Predicted scores/probabilities for positive class
            save_path: Path to save figure
        
        Returns:
            Path where figure was saved
        """
        if save_path is None:
            save_path = self.output_dir / "pr_curve.png"
        else:
            save_path = Path(save_path)
        
        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Plot PR curve
        ax.plot(recall, precision, color='darkblue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        
        # Random classifier baseline (depends on class balance)
        baseline_precision = (y_true == 1).sum() / len(y_true)
        ax.axhline(y=baseline_precision, color='gray', lw=2, linestyle='--', label=f'Baseline (P={baseline_precision:.4f})')
        
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ PR curve saved to {save_path}")
        return str(save_path)
    
    def plot_class_distribution(
        self,
        y: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Class Distribution",
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot class distribution in histogram/bar chart.
        
        Args:
            y: Labels
            labels: Class labels
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Path where figure was saved
        """
        if save_path is None:
            save_path = self.output_dir / "class_distribution.png"
        else:
            save_path = Path(save_path)
        
        if labels is None:
            labels = ['Normal', 'Suspicious']
        
        unique, counts = np.unique(y, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(range(len(unique)), counts, color=['#2ecc71', '#e74c3c'])
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels([labels[i] for i in unique])
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            percentage = 100 * count / counts.sum()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{int(count)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Class distribution saved to {save_path}")
        return str(save_path)
    
    def plot_all_results(
        self,
        history: Dict[str, Any],
        y_true_final: np.ndarray,
        y_pred_final: np.ndarray,
        y_scores_final: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate all visualizations at once.
        
        Args:
            history: Training history dict
            y_true_final: Final test set true labels
            y_pred_final: Final test set predictions
            y_scores_final: Final test set score probabilities (for ROC/PR)
            output_dir: Override output directory
        
        Returns:
            Dict mapping plot names to file paths
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            plots['loss_curves'] = self.plot_loss_curves(history)
        
        # Metrics curves
        if 'val_metrics' in history:
            plots['metrics_curves'] = self.plot_metrics_curves(history)
        
        # Confusion matrix
        plots['confusion_matrix'] = self.plot_confusion_matrix(y_true_final, y_pred_final)
        
        # ROC curve
        if y_scores_final is not None:
            plots['roc_curve'] = self.plot_roc_curve(y_true_final, y_scores_final)
            plots['pr_curve'] = self.plot_precision_recall_curve(y_true_final, y_scores_final)
        
        # Class distribution
        plots['class_distribution_test'] = self.plot_class_distribution(y_true_final, title="Test Set Class Distribution")
        
        return plots
    
    def generate_summary_report(
        self,
        history: Dict[str, Any],
        y_true_final: np.ndarray,
        y_pred_final: np.ndarray,
        y_scores_final: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a text summary report.
        
        Args:
            history: Training history
            y_true_final: Test labels
            y_pred_final: Test predictions
            y_scores_final: Test scores
            save_path: Path to save report
        
        Returns:
            Path where report was saved
        """
        if save_path is None:
            save_path = self.output_dir / "training_report.txt"
        else:
            save_path = Path(save_path)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        # Compute metrics
        accuracy = accuracy_score(y_true_final, y_pred_final)
        precision = precision_score(y_true_final, y_pred_final, zero_division=0)
        recall = recall_score(y_true_final, y_pred_final, zero_division=0)
        f1 = f1_score(y_true_final, y_pred_final, zero_division=0)
        
        auc_score = None
        if y_scores_final is not None:
            auc_score = roc_auc_score(y_true_final, y_scores_final)
        
        cm = confusion_matrix(y_true_final, y_pred_final)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Write report
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Training and Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("TRAINING HISTORY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Number of epochs: {len(history.get('train_loss', []))}\n")
            if 'train_loss' in history:
                f.write(f"Final train loss: {history['train_loss'][-1]:.6f}\n")
            if 'val_loss' in history:
                f.write(f"Final val loss: {history['val_loss'][-1]:.6f}\n")
                f.write(f"Best val loss: {min(history['val_loss']):.6f}\n")
            f.write("\n")
            
            f.write("TEST SET RESULTS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            if auc_score is not None:
                f.write(f"AUC-ROC:   {auc_score:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write("\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 60 + "\n")
            f.write(f"True Negatives:  {tn}\n")
            f.write(f"False Positives: {fp}\n")
            f.write(f"False Negatives: {fn}\n")
            f.write(f"True Positives:  {tp}\n")
            f.write("\n")
            
            f.write("CLASS DISTRIBUTION (TEST SET)\n")
            f.write("-" * 60 + "\n")
            unique, counts = np.unique(y_true_final, return_counts=True)
            for label, count in zip(unique, counts):
                percentage = 100 * count / len(y_true_final)
                f.write(f"Class {label}: {int(count)} ({percentage:.2f}%)\n")
        
        print(f"✓ Summary report saved to {save_path}")
        return str(save_path)
