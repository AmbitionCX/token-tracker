"""
Unified Trainer: Training loop for all model types (sklearn + PyTorch).

Supports:
- Mini-batch training
- Validation and testing
- Checkpointing
- Early stopping
- Metric computation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional, Callable, List, Tuple, Any
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import logging


class Trainer:
    """
    Unified trainer for PyTorch models.
    
    Handles training loop, validation, checkpointing, and metric tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        use_tensorboard: bool = True
    ):
        """
        Args:
            model: PyTorch model to train
            device: "cuda" or "cpu"
            checkpoint_dir: Where to save checkpoints
            log_dir: Where to save logs
            use_tensorboard: Whether to use TensorBoard
        """
        self.model = model.to(device)
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # TensorBoard
        if use_tensorboard:
            self.writer = SummaryWriter(str(self.log_dir))
        else:
            self.writer = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.best_epoch = 0
    
    def _setup_logging(self):
        """Setup logging."""
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / 'train.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def train_epoch(
        self,
        train_loader,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        grad_clip: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            grad_clip: Gradient clipping norm
        
        Returns:
            Dictionary with metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch} [Train]')
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Prepare batch - move tensors to device
            batch_device = {}
            labels = None
            
            for k, v in batch.items():
                if k == 'labels':
                    labels = v.to(self.device) if isinstance(v, torch.Tensor) else v
                elif k == 'num_edges':
                    # Skip num_edges - not needed by model forward
                    continue
                elif isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(self.device)
                else:
                    batch_device[k] = v
            
            # Forward pass
            try:
                logits = self.model(**batch_device)
                loss = loss_fn(logits, labels)
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Backward pass
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            avg_loss = total_loss / max(1, num_batches)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # TensorBoard logging
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
        
        return {'loss': total_loss / max(1, num_batches)}
    
    def validate(
        self,
        val_loader,
        loss_fn: Callable,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            metric_fn: Optional metric computation function
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.epoch} [Val]')
            
            for batch in pbar:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Extract labels and num_edges (not needed by model)
                labels = batch.pop('labels')
                batch.pop('num_edges', None)
                
                try:
                    logits = self.model(**batch)
                    loss = loss_fn(logits, labels)
                except Exception as e:
                    self.logger.error(f"Validation batch error: {e}")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{total_loss / max(1, num_batches):.4f}'})
        
        metrics = {'loss': total_loss / max(1, num_batches)}
        
        # Compute additional metrics
        if metric_fn and all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            extra_metrics = metric_fn(all_preds, all_labels)
            metrics.update(extra_metrics)
        
        return metrics
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        loss_fn: Optional[Callable] = None,
        metric_fn: Optional[Callable] = None,
        patience: int = 10,
        save_best: bool = True,
        scheduler: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization
            loss_fn: Loss function
            metric_fn: Metric computation
            patience: Early stopping patience
            save_best: Save best model
            scheduler: Learning rate scheduler type
        
        Returns:
            Training history
        """
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Setup scheduler
        if scheduler == 'cosine':
            scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        else:
            scheduler_obj = None
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        
        # Early stopping
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, loss_fn)
            history['train_loss'].append(train_metrics['loss'])
            self.logger.info(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader, loss_fn, metric_fn)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)
            self.logger.info(f"Epoch {epoch}: Val Loss={val_metrics['loss']:.4f}")
            
            # TensorBoard
            if self.writer:
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                for k, v in val_metrics.items():
                    if k != 'loss':
                        self.writer.add_scalar(f'val/{k}', v, epoch)
            
            # Learning rate scheduler step
            if scheduler_obj:
                scheduler_obj.step()
            
            # Early stopping and checkpointing
            metric_value = val_metrics.get('f1', val_metrics['loss'])
            
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                self.best_epoch = epoch
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(f'best_model.pt')
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info(f"Training completed. Best epoch: {self.best_epoch} with metric={self.best_metric:.4f}")
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state': self.model.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def close(self) -> None:
        """Cleanup."""
        if self.writer:
            self.writer.close()
