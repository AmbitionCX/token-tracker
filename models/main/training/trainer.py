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
        
        if num_batches == 0:
            raise RuntimeError(
                "No training batches were processed successfully. "
                "Check preceding batch processing errors for root cause."
            )
        return {'loss': total_loss / num_batches}
    
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
        
        if num_batches == 0:
            raise RuntimeError(
                "No validation batches were processed successfully. "
                "Check preceding validation batch errors for root cause."
            )
        metrics = {'loss': total_loss / num_batches}
        
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


# ============================================================================
# GNNWindowTrainer — 2-stage training loop for GNN-based models (models 5-8)
# ============================================================================

class GNNWindowTrainer:
    """
    Trainer for GNN-based models where each step processes a full temporal
    graph window.

    Memory strategy
    ---------------
    Each temporal window can contain tens of thousands of edges.  Running the
    Transformer trace encoder over all edges at once would exhaust GPU memory.
    The 2-stage loop solves this:

      Stage 1 – mini-batch trace encoding (chunk_size edges at a time, grads kept)
      Stage 2 – full-graph GNN on node features  (node count << edge count)
      Stage 3 – mini-batch edge classification with gradient accumulation

    All three stages share the same optimizer step, so end-to-end gradients
    flow through trace encoder → edge features → GNN → classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        chunk_size: int = 512,       # edges per mini-batch inside a window
    ):
        self.model = model.to(device)
        self.device = device
        self.chunk_size = chunk_size

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()

        if use_tensorboard:
            self.writer = SummaryWriter(str(self.log_dir))
        else:
            self.writer = None

        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.best_epoch = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _setup_logging(self):
        logger = logging.getLogger(f'GNNWindowTrainer_{id(self)}')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_dir / 'train.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return x

    def _chunked(self, total: int):
        """Yield (start, end) index pairs of size chunk_size."""
        for start in range(0, total, self.chunk_size):
            yield start, min(start + self.chunk_size, total)

    # ------------------------------------------------------------------
    # forward helpers that operate on index slices of a full-window batch
    # ------------------------------------------------------------------

    def _encode_traces_chunked(self, window: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run trace encoder over all edges in a window in mini-batches.

        Returns trace_repr (E, trace_hidden_dim) with gradients retained.
        """
        E = window['call_type_ids'].size(0)
        trace_reprs = []

        for s, e in self._chunked(E):
            chunk_emb = self.model.edge_feature_extractor.call_event_embedding(
                window['call_type_ids'][s:e].to(self.device),
                window['contract_ids'][s:e].to(self.device),
                window['func_selector_ids'][s:e].to(self.device),
                window['depths'][s:e].to(self.device),
                window['status_ids'][s:e].to(self.device),
                window['input_sizes'][s:e].to(self.device),
                window['output_sizes'][s:e].to(self.device),
                window['gas_vals'][s:e].to(self.device),
            )
            chunk_repr = self.model.trace_encoder(
                chunk_emb,
                window['trace_mask'][s:e].to(self.device)
            )  # (chunk, hidden_dim)
            trace_reprs.append(chunk_repr)

        return torch.cat(trace_reprs, dim=0)  # (E, hidden_dim)

    def _build_edge_features(
        self,
        window: Dict[str, torch.Tensor],
        trace_repr: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate external features (if enabled) and trace repr."""
        if self.model.use_external:
            ext = window['external_features'].to(self.device)  # (E, 4)
            return torch.cat([ext, trace_repr], dim=-1)         # (E, 4 + trace_hidden_dim)
        return trace_repr                                        # (E, 128)

    # ------------------------------------------------------------------
    # train / validate
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_windows = 0

        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch} [Train]')

        for window in pbar:
            E = window['call_type_ids'].size(0)
            if E == 0:
                continue

            optimizer.zero_grad()

            # ---- Stage 1: trace encoding (chunked, grads kept) ----
            if self.model.use_trace:
                trace_repr = self._encode_traces_chunked(window)  # (E, hidden_dim)
            else:
                trace_repr = torch.zeros(
                    E, self.model.edge_feature_extractor.trace_embedding_dim,
                    device=self.device
                )

            # ---- Stage 2: full-graph GNN ----
            edge_features = self._build_edge_features(window, trace_repr)  # (E, edge_dim)

            edge_index    = window['edge_index'].to(self.device)      # (2, E)
            node_features = window['node_features'].to(self.device)   # (N, D)

            if self.model.use_gnn:
                node_repr = self.model.gnn(node_features, edge_index, edge_features)
            else:
                node_repr = None

            # ---- Stage 3: chunked edge classification + gradient accumulation ----
            labels    = window['labels'].to(self.device)   # (E,)
            num_chunks = max(1, (E + self.chunk_size - 1) // self.chunk_size)
            window_loss = 0.0

            for chunk_i, (s, e) in enumerate(self._chunked(E)):
                is_last = (chunk_i == num_chunks - 1)

                ef_chunk = edge_features[s:e]  # (chunk, edge_dim)

                if node_repr is not None:
                    src_nodes = edge_index[0, s:e]
                    dst_nodes = edge_index[1, s:e]
                    h_u = node_repr[src_nodes]
                    h_v = node_repr[dst_nodes]
                else:
                    gnn_dim = self.model.gnn.hidden_dim if self.model.gnn else 128
                    h_u = torch.zeros(e - s, gnn_dim, device=self.device)
                    h_v = torch.zeros(e - s, gnn_dim, device=self.device)

                # Aggregation step
                if self.model.use_attention:
                    z_e = self.model.edge_aggregator(h_u, h_v, ef_chunk)
                else:
                    z_e = self.model.edge_aggregator(h_u, h_v, ef_chunk)

                logits = self.model.classifier(z_e)
                loss = loss_fn(logits, labels[s:e]) / num_chunks
                loss.backward(retain_graph=not is_last)
                window_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += window_loss
            num_windows += 1
            self.global_step += 1

            pbar.set_postfix({'loss': f'{total_loss / num_windows:.4f}'})
            if self.writer and self.global_step % 5 == 0:
                self.writer.add_scalar('train/loss', total_loss / num_windows, self.global_step)

        if num_windows == 0:
            raise RuntimeError("No windows processed in training epoch.")
        return {'loss': total_loss / num_windows}

    def validate(
        self,
        val_loader,
        loss_fn: Callable,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_windows = 0
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.epoch} [Val]')
            for window in pbar:
                E = window['call_type_ids'].size(0)
                if E == 0:
                    continue

                if self.model.use_trace:
                    trace_repr = self._encode_traces_chunked(window)
                else:
                    trace_repr = torch.zeros(
                        E, self.model.edge_feature_extractor.trace_embedding_dim,
                        device=self.device
                    )

                edge_features = self._build_edge_features(window, trace_repr)
                edge_index    = window['edge_index'].to(self.device)
                node_features = window['node_features'].to(self.device)
                labels        = window['labels'].to(self.device)

                if self.model.use_gnn:
                    node_repr = self.model.gnn(node_features, edge_index, edge_features)
                else:
                    node_repr = None

                win_logits = []
                for s, e in self._chunked(E):
                    ef_chunk = edge_features[s:e]
                    if node_repr is not None:
                        h_u = node_repr[edge_index[0, s:e]]
                        h_v = node_repr[edge_index[1, s:e]]
                    else:
                        gnn_dim = self.model.gnn.hidden_dim if self.model.gnn else 128
                        h_u = torch.zeros(e - s, gnn_dim, device=self.device)
                        h_v = torch.zeros(e - s, gnn_dim, device=self.device)

                    z_e    = self.model.edge_aggregator(h_u, h_v, ef_chunk)
                    logits = self.model.classifier(z_e)
                    win_logits.append(logits)

                win_logits = torch.cat(win_logits, dim=0)
                loss = loss_fn(win_logits, labels)
                total_loss += loss.item()
                num_windows += 1

                preds = torch.argmax(win_logits, dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                pbar.set_postfix({'loss': f'{total_loss / num_windows:.4f}'})

        if num_windows == 0:
            raise RuntimeError("No windows processed in validation.")

        metrics = {'loss': total_loss / num_windows}
        if metric_fn and all_preds:
            metrics.update(metric_fn(
                np.concatenate(all_preds),
                np.concatenate(all_labels)
            ))
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
        scheduler: Optional[str] = None,
    ) -> Dict[str, List]:
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        if scheduler == 'cosine':
            scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        else:
            scheduler_obj = None

        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        patience_counter = 0

        for epoch in range(num_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch(train_loader, optimizer, loss_fn)
            history['train_loss'].append(train_metrics['loss'])
            self.logger.info(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}")

            val_metrics = self.validate(val_loader, loss_fn, metric_fn)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)
            self.logger.info(f"Epoch {epoch}: Val Loss={val_metrics['loss']:.4f}")

            if self.writer:
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                for k, v in val_metrics.items():
                    if k != 'loss' and isinstance(v, (int, float)):
                        self.writer.add_scalar(f'val/{k}', v, epoch)

            if scheduler_obj:
                scheduler_obj.step()

            metric_value = val_metrics.get('macro_f1', val_metrics.get('f1', val_metrics['loss']))
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                self.best_epoch = epoch
                patience_counter = 0
                if save_best:
                    self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        self.logger.info(f"Training done. Best epoch {self.best_epoch}, metric={self.best_metric:.4f}")
        return history

    def save_checkpoint(self, filename: str) -> None:
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state': self.model.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def close(self) -> None:
        if self.writer:
            self.writer.close()
