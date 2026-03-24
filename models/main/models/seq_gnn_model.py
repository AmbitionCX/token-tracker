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
from .gnn_base import GNNBase, NodeInitMLP


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
        external_dim: int = 4,
        use_external: bool = True,   # False → "Trace only" models
        
        # Trace encoding config
        trace_encoder_type: str = "transformer",  # transformer, lstm, pooling
        trace_hidden_dim: int = 128,
        trace_num_layers: int = 2,
        trace_num_heads: int = 4,
        
        # Edge feature config (computed automatically; kept for backward compat)
        edge_dim: Optional[int] = None,  # if None, auto = (external_dim if use_external else 0) + trace_hidden_dim
        
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
        self.use_external = use_external
        self.use_trace = use_trace
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        
        # Edge dim auto-computed: external part (0 when use_external=False) + trace part
        effective_edge_dim = (external_dim if use_external else 0) + trace_hidden_dim
        self.edge_dim = edge_dim if edge_dim is not None else effective_edge_dim
        edge_dim = self.edge_dim
        
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
            trace_input_dim = self.edge_feature_extractor.call_event_embedding.output_dim
            self.trace_encoder = TraceEncoder(
                input_dim=trace_input_dim,
                hidden_dim=trace_hidden_dim,
                encoder_type=trace_encoder_type,
                num_layers=trace_num_layers,
                num_heads=trace_num_heads,
                dropout=dropout
            )
        
        # ===== MODULE 2: Node initialisation + GNN =====
        # s_v = [log(1+total_tx), log(1+in_deg), log(1+out_deg)]  (3-dim)
        node_stat_dim = 3
        node_init_dim = gnn_hidden_dim  # h_v^(0) ∈ R^{gnn_hidden_dim}

        self.node_init_mlp = NodeInitMLP(
            stat_dim=node_stat_dim,
            hidden_dim=gnn_hidden_dim // 2,
            output_dim=node_init_dim,
        )

        if use_gnn:
            self.gnn = GNNBase(
                input_dim=node_init_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_hidden_dim,
                num_layers=gnn_num_layers,
                gnn_type=gnn_type,
                num_heads=gnn_num_heads,
                dropout=dropout,
                edge_dim=edge_dim if gnn_type == "gat" else None,
            )
        else:
            self.gnn = None
        
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
        external_features: torch.Tensor,        # (batch_size, 4)
        
        # Internal trace tokenized data
        call_type_ids: torch.Tensor,            # (batch_size, seq_len)
        contract_ids: torch.Tensor,             # (batch_size, seq_len)
        func_selector_ids: torch.Tensor,        # (batch_size, seq_len)
        depths: torch.Tensor,                   # (batch_size, seq_len)
        status_ids: torch.Tensor,               # (batch_size, seq_len) 0/1
        input_sizes: torch.Tensor,              # (batch_size, seq_len) log(1+len)
        output_sizes: torch.Tensor,             # (batch_size, seq_len)
        gas_vals: torch.Tensor,                # (batch_size, seq_len)
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
        
        # When use_external=False, external_features is still passed in (ignored for edge features)
        # but used to infer batch_size/device. Callers may pass a zero tensor of correct shape.
        
        # ===== STEP 1: Extract and encode trace features =====
        if self.use_trace:
            # Get call event embeddings from trace encoder
            trace_emb = self.edge_feature_extractor.call_event_embedding(
                call_type_ids,
                contract_ids,
                func_selector_ids,
                depths,
                status_ids,
                input_sizes,
                output_sizes,
                gas_vals,
                trace_mask,
            )  # (batch_size, seq_len, call_event_embedding.output_dim)
            # Zero padding positions before the Transformer so attention backward does not
            # inject unstable gradients into embedding / linear rows used only for pad tokens.
            if trace_mask is not None:
                trace_emb = trace_emb * trace_mask.unsqueeze(-1).float()
            # TraceEncoder's internal input_proj handles the projection to hidden_dim
            
            # Encode trace sequence
            trace_repr = self.trace_encoder(trace_emb, trace_mask)
            # (batch_size, trace_hidden_dim)
        else:
            trace_repr = torch.zeros(batch_size, self.edge_feature_extractor.trace_embedding_dim, device=device)
        
        # ===== STEP 2: Combine external + trace features =====
        if self.use_external:
            edge_features = torch.cat([external_features, trace_repr], dim=-1)
        else:
            edge_features = trace_repr  # "Trace only" models: skip external features
        
        # ===== STEP 3: Node initialisation + GNN =====
        # h_v^(0) = NodeInitMLP(s_v)  where s_v ∈ R^3
        if node_features is not None:
            node_h0 = self.node_init_mlp(node_features.to(device))  # (N, gnn_hidden_dim)
        else:
            node_h0 = None

        if self.use_gnn and node_h0 is not None and edge_index is not None:
            # h_v^(K) via multi-layer edge-aware GAT message passing
            node_repr = self.gnn(node_h0, edge_index, edge_features)
            # (N, gnn_hidden_dim)

            # Retrieve h_u, h_v for edges in this batch
            if edge_indices_in_batch is not None:
                src = edge_index[0, edge_indices_in_batch]
                dst = edge_index[1, edge_indices_in_batch]
            else:
                src = edge_index[0]
                dst = edge_index[1]
            h_u = node_repr[src]  # (batch_size, gnn_hidden_dim)
            h_v = node_repr[dst]
        else:
            gnn_dim = self.gnn.hidden_dim if self.gnn else 128
            h_u = torch.zeros(batch_size, gnn_dim, device=device)
            h_v = torch.zeros(batch_size, gnn_dim, device=device)
        
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
