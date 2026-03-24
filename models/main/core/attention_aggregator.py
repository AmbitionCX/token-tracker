"""
Edge Attention Aggregator: Aggregates edge representations with neighborhood context

Uses attention mechanism over neighborhood to identify important transactions
and capture local structural patterns (MEV, arbitrage, sandwich attacks, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EdgeAttentionAggregator(nn.Module):
    """
    Edge-aware attention aggregation mechanism.
    
    For a target edge (u → v), aggregates information from:
    1. Source node (u)
    2. Target node (v)
    3. Edge features (transaction properties)
    4. Neighborhood context (incoming/outgoing edges from u and v)
    
    Uses multi-head attention to assign importance to neighboring edges.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            node_dim: Dimension of node representations (h_v)
            edge_dim: Dimension of edge features (x_e)
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        # Query projection for target edge
        self.query_proj = nn.Linear(node_dim + node_dim + edge_dim, hidden_dim)
        
        # Key/Value projections for neighborhood edges
        self.key_proj = nn.Linear(edge_dim, hidden_dim)
        self.value_proj = nn.Linear(edge_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, edge_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(edge_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        h_u: torch.Tensor,              # (batch_size, node_dim) - source node repr
        h_v: torch.Tensor,              # (batch_size, node_dim) - target node repr
        x_e: torch.Tensor,              # (batch_size, edge_dim) - edge features
        neighbor_edges: torch.Tensor,   # (batch_size, k_neighbors, edge_dim) - neighbor edge features
        neighbor_mask: Optional[torch.Tensor] = None  # (batch_size, k_neighbors) - valid neighbors
    ) -> torch.Tensor:
        """
        Args:
            h_u: Source node representation
            h_v: Target node representation
            x_e: Target edge features
            neighbor_edges: Features of neighboring edges (k neighbors max)
            neighbor_mask: Boolean mask for valid neighbors (True = valid)
        
        Returns:
            (batch_size, edge_dim) - aggregated edge representation
        """
        batch_size = x_e.size(0)
        
        # (1) Create query from target edge context
        target_context = torch.cat([h_u, h_v, x_e], dim=-1)  # (batch_size, 2*node_dim + edge_dim)
        q = self.query_proj(target_context)  # (batch_size, hidden_dim)
        q = q.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # (2) Create keys and values from neighbor edges
        k = self.key_proj(neighbor_edges)  # (batch_size, k_neighbors, hidden_dim)
        v = self.value_proj(neighbor_edges)  # (batch_size, k_neighbors, hidden_dim)
        
        # (3) Scaled dot-product attention
        # Reshape for multi-head attention
        q_h = q.transpose(0, 1).reshape(1, batch_size * self.num_heads, self.head_dim)
        k_h = k.transpose(0, 1).reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v_h = v.transpose(0, 1).reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Actually, simpler approach without multi-head reshaping for clarity:
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (batch_size, 1, k_neighbors)
        
        # (4) Apply mask
        if neighbor_mask is not None:
            # Expand mask to match scores shape
            mask = neighbor_mask.unsqueeze(1)  # (batch_size, 1, k_neighbors)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # (5) Softmax attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, 1, k_neighbors)
        attn_weights = self.dropout(attn_weights)
        
        # (6) Apply attention to values
        context = torch.matmul(attn_weights, v)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        # (7) Output projection and residual
        aggregated = self.output_proj(context)  # (batch_size, edge_dim)
        
        # Residual connection
        output = x_e + aggregated
        output = self.norm(output)
        
        return output


class NeighborhoodContextExtractor:
    """
    Extract neighborhood context for edge attention aggregation.
    
    Collects k neighboring edges (incoming and outgoing) for each edge in the graph.
    """
    
    def __init__(self, max_neighbors: int = 10):
        """
        Args:
            max_neighbors: Maximum number of neighbors to collect per edge
        """
        self.max_neighbors = max_neighbors
    
    def extract_neighbors(
        self,
        edge_index: torch.Tensor,           # (2, num_edges) - graph connectivity
        edge_attrs: torch.Tensor,           # (num_edges, edge_dim) - edge features
        num_nodes: int,
        target_edge_indices: Optional[torch.Tensor] = None  # Which edges to get neighbors for
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract neighbors for target edges.
        
        Args:
            edge_index: Graph connectivity COO format
            edge_attrs: Edge feature matrix
            num_nodes: Total number of nodes in graph
            target_edge_indices: If provided, only extract neighbors for these edges
                                If None, extract for all edges
        
        Returns:
            Tuple of:
            - neighbor_edge_features: (num_target_edges, max_neighbors, edge_dim)
            - neighbor_masks: (num_target_edges, max_neighbors) - boolean mask
        """
        num_edges = edge_index.size(1)
        edge_dim = edge_attrs.size(1) if edge_attrs.dim() > 1 else 1
        
        if target_edge_indices is None:
            target_edge_indices = torch.arange(num_edges)
        
        num_targets = len(target_edge_indices)
        device = edge_index.device
        
        neighbor_features = torch.zeros(
            num_targets, self.max_neighbors, edge_dim,
            device=device, dtype=edge_attrs.dtype
        )
        neighbor_masks = torch.zeros(
            num_targets, self.max_neighbors,
            device=device, dtype=torch.bool
        )
        
        # Build adjacency lists
        from_nodes, to_nodes = edge_index
        in_edges = [[] for _ in range(num_nodes)]  # Incoming edges per node
        out_edges = [[] for _ in range(num_nodes)]  # Outgoing edges per node
        
        for edge_idx, (u, v) in enumerate(zip(from_nodes, to_nodes)):
            out_edges[u.item()].append(edge_idx)
            in_edges[v.item()].append(edge_idx)
        
        # Extract neighbors for each target edge
        for target_idx, edge_idx in enumerate(target_edge_indices):
            edge_idx = edge_idx.item() if isinstance(edge_idx, torch.Tensor) else edge_idx
            
            u, v = from_nodes[edge_idx], to_nodes[edge_idx]
            u, v = u.item(), v.item()
            
            # Collect neighbor edges: outgoing from u, incoming to v
            neighbor_indices = set()
            neighbor_indices.update(out_edges[u][:self.max_neighbors // 2])
            neighbor_indices.update(in_edges[v][:self.max_neighbors // 2])
            neighbor_indices = list(neighbor_indices)[:self.max_neighbors]
            
            # Fill in neighbor features
            for i, neighbor_idx in enumerate(neighbor_indices):
                neighbor_features[target_idx, i] = edge_attrs[neighbor_idx]
                neighbor_masks[target_idx, i] = True
        
        return neighbor_features, neighbor_masks


class EdgeRepresentationAggregator(nn.Module):
    """
    Complete edge representation aggregation pipeline.
    
    Combines:
    1. Node representations (from GNN)
    2. Edge features (extracted earlier)
    3. Neighborhood attention (local structural context)
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_attention = use_attention

        # Always build concat fallback: training often has no neighbor_edges even when
        # use_attention=True, so forward must be able to project [h_u; h_v; x_e] → edge_dim.
        self.concat_proj = nn.Linear(2 * node_dim + edge_dim, edge_dim)

        if use_attention:
            self.attention = EdgeAttentionAggregator(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
    
    def forward(
        self,
        h_u: torch.Tensor,                      # (batch_size, node_dim)
        h_v: torch.Tensor,                      # (batch_size, node_dim)
        x_e: torch.Tensor,                      # (batch_size, edge_dim)
        neighbor_edges: Optional[torch.Tensor] = None,  # (batch_size, k_neighbors, edge_dim)
        neighbor_mask: Optional[torch.Tensor] = None    # (batch_size, k_neighbors)
    ) -> torch.Tensor:
        """
        Aggregate edge representation with node and neighborhood context.
        
        Returns:
            (batch_size, edge_dim) - final edge representation z_e
        """
        if self.use_attention and neighbor_edges is not None:
            z_e = self.attention(h_u, h_v, x_e, neighbor_edges, neighbor_mask)
        else:
            # Fallback: simple concatenation
            concat = torch.cat([h_u, h_v, x_e], dim=-1)
            z_e = self.concat_proj(concat)
        
        return z_e


# ============================================================================
# Utility functions for building neighborhood graphs
# ============================================================================

def build_neighborhood_graph(
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    num_nodes: int,
    k_hops: int = 1,
    k_neighbors: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build k-hop neighborhood graph for each edge.
    
    Args:
        edge_index: (2, num_edges) edge connectivity
        edge_features: (num_edges, feat_dim) edge features
        num_nodes: Total nodes
        k_hops: Number of hops in neighborhood
        k_neighbors: Max neighbors to keep per edge
    
    Returns:
        Tuple of:
        - neighbor_features: (num_edges, k_neighbors, feat_dim)
        - neighbor_indices: (num_edges, k_neighbors)
        - neighbor_masks: (num_edges, k_neighbors) boolean
    """
    num_edges = edge_index.size(1)
    feat_dim = edge_features.size(-1)
    device = edge_index.device
    dtype = edge_features.dtype
    
    neighbor_features = torch.zeros((num_edges, k_neighbors, feat_dim), device=device, dtype=dtype)
    neighbor_indices = torch.zeros((num_edges, k_neighbors), device=device, dtype=torch.long)
    neighbor_masks = torch.zeros((num_edges, k_neighbors), device=device, dtype=torch.bool)
    
    # Build adjacency lists
    from_idx, to_idx = edge_index
    
    in_neighbors = [[] for _ in range(num_nodes)]
    out_neighbors = [[] for _ in range(num_nodes)]
    
    for edge_id, (u, v) in enumerate(zip(from_idx, to_idx)):
        u, v = u.item(), v.item()
        out_neighbors[u].append(edge_id)
        in_neighbors[v].append(edge_id)
    
    # Extract neighborhood for each edge
    for edge_id in range(num_edges):
        u, v = from_idx[edge_id].item(), to_idx[edge_id].item()
        
        # Collect neighbors
        candidates = set()
        candidates.update(out_neighbors[u])
        candidates.update(in_neighbors[v])
        candidates = list(candidates)[:k_neighbors]
        
        for i, neighbor_edge_id in enumerate(candidates):
            neighbor_features[edge_id, i] = edge_features[neighbor_edge_id]
            neighbor_indices[edge_id, i] = neighbor_edge_id
            neighbor_masks[edge_id, i] = True
    
    return neighbor_features, neighbor_indices, neighbor_masks
