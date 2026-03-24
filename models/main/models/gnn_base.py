"""
GNN Base Module: Graph Neural Network for node representation learning

Implements different GNN architectures (GCN, GraphSAGE, GAT) for learning
node representations that capture address behavior in the transaction graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional, Tuple
import math


class GATConvLayer(nn.Module):
    """Graph Attention Network (GAT) layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        edge_dim: Optional[int] = None
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension (per head if concat=True)
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: If True, concatenate heads; else average
            edge_dim: Dimension of edge features (for GAT with edge attention)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        if out_features % num_heads != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        
        self.head_dim = out_features // num_heads
        
        # Linear transformations
        self.lin = nn.Linear(in_features, num_heads * self.head_dim, bias=False)
        
        # Attention weights
        self.att_l = Parameter(torch.Tensor(num_heads, self.head_dim, 1))
        self.att_r = Parameter(torch.Tensor(num_heads, self.head_dim, 1))
        
        # Edge features (if provided)
        self.lin_edge = None
        if edge_dim:
            self.lin_edge = nn.Linear(edge_dim, num_heads * self.head_dim, bias=False)
        
        # Bias and normalization
        if concat:
            self.bias = Parameter(torch.Tensor(num_heads * self.head_dim))
        else:
            self.bias = Parameter(torch.Tensor(self.head_dim))
        
        self.reset_parameters()
        self.dropout_layer = nn.Dropout(dropout)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.zeros_(self.bias)
        if self.lin_edge:
            nn.init.xavier_uniform_(self.lin_edge.weight)
    
    def forward(
        self,
        x: torch.Tensor,                    # (num_nodes, in_features)
        edge_index: torch.Tensor,          # (2, num_edges)
        edge_attr: Optional[torch.Tensor] = None  # (num_edges, edge_dim)
    ) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix
            edge_index: COO edge indices
            edge_attr: Edge feature matrix (optional)
        
        Returns:
            (num_nodes, out_features) node representations
        """
        num_nodes = x.size(0)
        
        # Project nodes to multiple heads
        x = self.lin(x)  # (num_nodes, num_heads * head_dim)
        x = x.view(-1, self.num_heads, self.head_dim)  # (num_nodes, num_heads, head_dim)
        
        # Compute attention coefficients
        from_idx, to_idx = edge_index
        
        # Attention scores from source nodes
        att_l = torch.einsum('nhd,ndl->nhl', x, self.att_l)  # (num_nodes, num_heads, 1)
        att_l = att_l[from_idx]  # (num_edges, num_heads, 1)
        
        # Attention scores from target nodes
        att_r = torch.einsum('nhd,ndl->nhl', x, self.att_r)
        att_r = att_r[to_idx]  # (num_edges, num_heads, 1)
        
        # Combine
        att = att_l + att_r  # (num_edges, num_heads, 1)
        
        # Add edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr)  # (num_edges, num_heads * head_dim)
            edge_feat = edge_feat.view(-1, self.num_heads, self.head_dim)
            att = att + torch.sum(
                x[from_idx] * edge_feat, dim=-1, keepdim=True
            ) / (self.head_dim ** 0.5)
        
        # Apply LeakyReLU and softmax
        att = F.leaky_relu(att, negative_slope=0.2)
        att = att.squeeze(-1)  # (num_edges, num_heads)
        
        # Softmax normalization per node
        att = self._softmax(att, to_idx, num_nodes)
        att = self.dropout_layer(att)
        
        # Apply attention to values
        x_j = x[from_idx]  # (num_edges, num_heads, head_dim)
        out = torch.einsum('eh,ehd->nhd', att, x_j)  # (num_nodes, num_heads, head_dim)
        
        if self.concat:
            out = out.contiguous().view(-1, self.num_heads * self.head_dim)
        else:
            out = out.mean(dim=1)
        
        out = out + self.bias
        
        return out
    
    @staticmethod
    def _softmax(att: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Softmax normalization for attention."""
        # Convert to sparse tensor for efficient softmax
        mask = torch.zeros((num_nodes, att.size(0)), device=att.device, dtype=torch.bool)
        mask[index] = True
        
        # Max value per node for numerical stability
        max_per_node = torch.full((num_nodes, att.size(1)), float('-inf'), device=att.device)
        max_per_node.scatter_(0, index.unsqueeze(-1).expand(-1, att.size(1)), 
                               torch.max(att, dim=0)[0].unsqueeze(0))
        
        att_exp = torch.exp(att - max_per_node[index])
        sum_per_node = torch.zeros((num_nodes, att.size(1)), device=att.device)
        sum_per_node.scatter_add_(0, index.unsqueeze(-1).expand(-1, att.size(1)), att_exp)
        
        return att_exp / (sum_per_node[index] + 1e-9)


class GCNConvLayer(nn.Module):
    """Graph Convolutional Network (GCN) layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_features)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional edge weights
        
        Returns:
            (num_nodes, out_features)
        """
        device = x.device
        num_nodes = x.size(0)
        
        # Compute normalization (D^{-1/2} A D^{-1/2})
        from_idx, to_idx = edge_index
        
        # Add self-loops
        self_loop = torch.arange(num_nodes, device=device).unsqueeze(0).expand(2, -1)
        edge_index_with_loops = torch.cat([edge_index, self_loop], dim=1)
        
        # Compute degree
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, to_idx, torch.ones_like(to_idx, dtype=torch.float))
        deg.scatter_add_(0, torch.arange(num_nodes, device=device), torch.ones(num_nodes, device=device))
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Messages
        x = self.lin(x)
        x = self.dropout(x)
        
        # Aggregation with normalization
        from_idx_loops, to_idx_loops = edge_index_with_loops
        messages = x[from_idx_loops] * deg_inv_sqrt[from_idx_loops].unsqueeze(-1) * deg_inv_sqrt[to_idx_loops].unsqueeze(-1)
        
        out = torch.zeros_like(x)
        out.scatter_add_(0, to_idx_loops.unsqueeze(-1).expand_as(messages), messages)
        
        return out


class GNNBase(nn.Module):
    """
    Graph Neural Network base module for node representation learning.
    
    Supports multiple GNN architectures: GAT (default), GCN, GraphSAGE.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        gnn_type: str = "gat",
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None
    ):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output (final) dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gat', 'gcn', 'graphsage')
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            edge_dim: Edge feature dimension (for GAT with edge attention)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        if gnn_type not in ["gat", "gcn", "graphsage"]:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Build layers
        self.layers = nn.ModuleList()
        
        if gnn_type == "gat":
            # GAT layers
            for i in range(num_layers):
                in_feat = input_dim if i == 0 else hidden_dim
                out_feat = output_dim if i == num_layers - 1 else hidden_dim
                
                self.layers.append(
                    GATConvLayer(
                        in_feat, out_feat,
                        num_heads=num_heads,
                        dropout=dropout,
                        concat=(i < num_layers - 1),  # Concat all but last
                        edge_dim=edge_dim
                    )
                )
        
        elif gnn_type == "gcn":
            # GCN layers
            for i in range(num_layers):
                in_feat = input_dim if i == 0 else hidden_dim
                out_feat = output_dim if i == num_layers - 1 else hidden_dim
                
                self.layers.append(
                    GCNConvLayer(in_feat, out_feat, dropout=dropout)
                )
        
        # Activation
        self.activation = F.relu
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN layers.
        
        Args:
            x: (num_nodes, input_dim) node features
            edge_index: (2, num_edges) COO edge indices
            edge_attr: (num_edges, edge_dim) optional edge features
        
        Returns:
            (num_nodes, output_dim) node representations
        """
        for i, layer in enumerate(self.layers):
            if self.gnn_type == "gat":
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)
            
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
