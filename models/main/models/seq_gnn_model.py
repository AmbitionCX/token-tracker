"""
SequenceGNN Model: Main proposed model combining sequence encoding + GNN + attention

Architecture:
1. External Feature Extraction
2. Internal Call Trace Encoding (Transformer/LSTM/Pooling)
3. Edge Feature Combination
4. Graph Neural Network (GAT/GCN/GraphSAGE) for node representation
5. Edge-aware Attention Aggregation
6. Binary Classification (Suspicious vs Normal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
import os

# Import from core modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

from core import (
    TraceEncoder,
    EdgeFeatureExtractor,
    EdgeRepresentationAggregator,
    build_neighborhood_graph
)
from .gnn_base import GNNBase


class SequenceGNNModel(nn.Module):
    """
    Complete Sequence+GNN model for suspicious transaction detection.
    
    Combines:
    - Internal call trace encoding (sequence modeling)
    - GNN for node representation learning
    - Neighborhood attention for edge classification
    """
    
    def __init__(
        self,
        # External feature config
        external_dim: int = 7,
        
        # Trace encoding config
        trace_encoder_type: str = "transformer",  # transformer, lstm, pooling
        trace_hidden_dim: int = 128,
        trace_num_layers: int = 2,
        trace_num_heads: int = 4,
        
        # Edge feature config
        edge_dim: int = 135,  # 7 external + 128 trace
        
        # GNN config
        gnn_type: str = "gat",  # gat, gcn, graphsage
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        
        # Attention aggregation config
        attn_hidden_dim: int = 64,
        attn_num_heads: int = 4,
        use_attention: bool = True,
        
        # Classifier config
        classifier_hidden_dim: int = 64,
        num_classes: int = 2,
        
        # General config
        dropout: float = 0.1,
        use_trace: bool = True,
        use_gnn: bool = True
    ):
        super().__init__()
        
        self.external_dim = external_dim
        self.edge_dim = edge_dim
        self.use_trace = use_trace
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        
        # ===== MODULE 1: External + Trace Feature Extraction =====
        # Edge feature extractor (handles external + trace features)
        self.edge_feature_extractor = EdgeFeatureExtractor(
            trace_embedding_dim=trace_hidden_dim,
            max_trace_length=256,
            use_trace=use_trace,
            use_external=True
        )
        
        # Trace encoder (if enabled)
        if use_trace:
            self.trace_encoder = TraceEncoder(
                input_dim=trace_hidden_dim,  # 5 embeddings × (128/5) = 128
                hidden_dim=trace_hidden_dim,
                encoder_type=trace_encoder_type,
                num_layers=trace_num_layers,
                num_heads=trace_num_heads,
                dropout=dropout
            )
        
        # ===== MODULE 2: GNN for Node Representation =====
        # Initialize GNN (node features TBD at inference time)
        node_initial_feat_dim = 10  # Placeholder (EOA/contract type, etc.)
        
        if use_gnn:
            self.gnn = GNNBase(
                input_dim=node_initial_feat_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_hidden_dim,
                num_layers=gnn_num_layers,
                gnn_type=gnn_type,
                num_heads=gnn_num_heads,
                dropout=dropout,
                edge_dim=edge_dim if gnn_type == "gat" else None
            )
        else:
            # No GNN: use dummy node representations
            self.gnn = None
            self.dummy_node_proj = nn.Linear(node_initial_feat_dim, gnn_hidden_dim)
        
        # ===== MODULE 3: Edge Attention Aggregation =====
        if use_attention:
            self.edge_aggregator = EdgeRepresentationAggregator(
                node_dim=gnn_hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=attn_hidden_dim,
                num_heads=attn_num_heads,
                dropout=dropout,
                use_attention=True
            )
        else:
            # Simple concatenation baseline
            self.edge_aggregator = EdgeRepresentationAggregator(
                node_dim=gnn_hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=attn_hidden_dim,
                num_heads=attn_num_heads,
                dropout=dropout,
                use_attention=False
            )
        
        # ===== MODULE 4: Classification Head =====
        self.classifier = nn.Sequential(
            nn.Linear(edge_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        # External features
        external_features: torch.Tensor,        # (batch_size, 7)
        
        # Internal trace tokenized data
        call_type_ids: torch.Tensor,            # (batch_size, seq_len)
        contract_ids: torch.Tensor,             # (batch_size, seq_len)
        func_selector_ids: torch.Tensor,        # (batch_size, seq_len)
        depths: torch.Tensor,                   # (batch_size, seq_len)
        exec_properties: torch.Tensor,          # (batch_size, seq_len, 4)
        trace_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
        
        # Graph structure
        edge_index: Optional[torch.Tensor] = None,      # (2, num_edges) in batch
        node_features: Optional[torch.Tensor] = None,   # (num_nodes_in_batch, node_feat_dim)
        num_nodes: Optional[int] = None,                # Total nodes in batch
        
        # Edge indices within batch
        edge_indices_in_batch: Optional[torch.Tensor] = None,  # (num_edges,) indices in batch
        
        # Neighborhood context
        neighbor_edges: Optional[torch.Tensor] = None,  # (num_edges, k_neighbors, edge_dim)
        neighbor_mask: Optional[torch.Tensor] = None    # (num_edges, k_neighbors)
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            external_features: Extracted external transaction features
            call_type_ids, etc.: Tokenized internal call trace data
            edge_index: Graph connectivity
            node_features: Node feature matrix
            edge_indices_in_batch: Which edges are in the batch
            neighbor_edges, neighbor_mask: Neighborhood context for edges
        
        Returns:
            (batch_size, num_classes) - logits
        """
        batch_size = external_features.size(0)
        device = external_features.device
        
        # ===== STEP 1: Extract and encode trace features =====
        if self.use_trace:
            # Get call event embeddings from trace encoder
            trace_emb = self.edge_feature_extractor.call_event_embedding(
                call_type_ids, contract_ids, func_selector_ids, depths, exec_properties
            )  # (batch_size, seq_len, trace_embedding_dim)
            
            # Encode trace sequence
            trace_repr = self.trace_encoder(trace_emb, trace_mask)
            # (batch_size, trace_hidden_dim)
        else:
            trace_repr = torch.zeros(batch_size, self.edge_feature_extractor.trace_embedding_dim, device=device)
        
        # ===== STEP 2: Combine external + trace features =====
        # Simple concatenation
        edge_features = torch.cat([external_features, trace_repr], dim=-1)
        # (batch_size, edge_dim=135)
        
        # ===== STEP 3: Graph Neural Network (optional) =====
        if self.use_gnn and node_features is not None and edge_index is not None:
            # Forward through GNN
            node_repr = self.gnn(node_features, edge_index, edge_features)
            # (num_nodes_in_batch, gnn_hidden_dim)
            
            # Get source and target node representations for edges in batch
            if edge_indices_in_batch is not None:
                edge_src_idx = edge_index[0, edge_indices_in_batch]
                edge_dst_idx = edge_index[1, edge_indices_in_batch]
                h_u = node_repr[edge_src_idx]  # (batch_size, gnn_hidden_dim)
                h_v = node_repr[edge_dst_idx]  # (batch_size, gnn_hidden_dim)
            else:
                # Fallback: assume edges are already in batch order
                h_u = torch.zeros(batch_size, node_repr.size(-1), device=device)
                h_v = torch.zeros(batch_size, node_repr.size(-1), device=device)
        else:
            # No GNN: use dummy representations from initial node features
            gnn_hidden_dim = self.gnn.hidden_dim if self.gnn else 128
            h_u = torch.zeros(batch_size, gnn_hidden_dim, device=device)
            h_v = torch.zeros(batch_size, gnn_hidden_dim, device=device)
        
        # ===== STEP 4: Edge Attention Aggregation (optional) =====
        if self.use_attention and neighbor_edges is not None:
            z_e = self.edge_aggregator(h_u, h_v, edge_features, neighbor_edges, neighbor_mask)
            # (batch_size, edge_dim)
        else:
            # No attention: direct combination
            z_e = self.edge_aggregator(h_u, h_v, edge_features)
            # (batch_size, edge_dim)
        
        # ===== STEP 5: Classification =====
        logits = self.classifier(z_e)  # (batch_size, num_classes)
        
        return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        positive_weight: float = 10.0
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss for imbalanced classification.
        
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,) - 0/1 labels
            positive_weight: Weight for positive class (suspicious)
        
        Returns:
            Scalar loss
        """
        weights = torch.where(labels == 1, positive_weight, 1.0)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, positive_weight], device=logits.device))
        loss = loss_fn(logits, labels)
        return loss


# ============================================================================
# Ablation Model Variants
# ============================================================================

class NoTraceSequenceGNN(SequenceGNNModel):
    """Ablation: Remove trace features (use only external features)."""
    
    def __init__(self, **kwargs):
        kwargs['use_trace'] = False
        super().__init__(**kwargs)


class NoGNNSequenceModel(SequenceGNNModel):
    """Ablation: Remove GNN (direct edge classification from features)."""
    
    def __init__(self, **kwargs):
        kwargs['use_gnn'] = False
        super().__init__(**kwargs)


class NoAttentionSequenceGNN(SequenceGNNModel):
    """Ablation: Remove attention (simple concatenation)."""
    
    def __init__(self, **kwargs):
        kwargs['use_attention'] = False
        super().__init__(**kwargs)


class LSTMSequenceGNN(SequenceGNNModel):
    """Ablation: Use LSTM instead of Transformer for trace encoding."""
    
    def __init__(self, **kwargs):
        kwargs['trace_encoder_type'] = 'lstm'
        super().__init__(**kwargs)


class PoolingSequenceGNN(SequenceGNNModel):
    """Ablation: Use simple pooling instead of Transformer for trace encoding."""
    
    def __init__(self, **kwargs):
        kwargs['trace_encoder_type'] = 'pooling'
        super().__init__(**kwargs)
