"""
GNN Base Module: Node initialisation + Graph Attention Network

Implements §5 of the paper:
  - NodeInitMLP  : h_v^(0) = MLP(s_v)   where s_v = [total_tx, in_deg, out_deg]
  - EdgeAwareGATLayer : message passing with edge features in attention weight
      h_v^(k+1) = σ( Σ_{u∈N(v)} α_{uv}^(k) · W^(k) [h_u^(k) ; x_{u→v}] )
      α_{uv}^(k) = softmax_u( LeakyReLU( a^T [h_u ; h_v ; x_{u→v}] ) )
  - GNNBase : multi-layer wrapper (GAT / GCN / GraphSAGE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional
import math


# ===========================================================================
# §5.1 Node initialisation
# ===========================================================================

class NodeInitMLP(nn.Module):
    """
    Maps per-address statistics s_v to an initial node embedding h_v^(0).

    s_v = [total_tx_count, in_degree, out_degree]  — 3-dim log-normalised vector
    h_v^(0) = MLP(s_v)  = LayerNorm( ReLU( W_2 · ReLU( W_1 · s_v ) ) )
    """

    def __init__(self, stat_dim: int = 3, hidden_dim: int = 32, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, node_stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_stats: (N, stat_dim) — log(1+x) normalised statistics
        Returns:
            (N, output_dim)
        """
        return self.net(node_stats)


# ===========================================================================
# §5.1  Edge-aware GAT layer
# ===========================================================================

class EdgeAwareGATLayer(nn.Module):
    """
    Single GAT layer that incorporates edge features in both the message and
    the attention coefficient (matches the paper's formulation exactly).

    Message :  W^(k) · [h_u^(k) ; x_{u→v}]
    Attention: a^T · LeakyReLU( W_a · [h_u ; h_v ; x_{u→v}] )
    """

    def __init__(
        self,
        node_in: int,
        node_out: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)

        # concat=True : output = num_heads * head_dim = node_out  → head_dim = node_out // num_heads
        # concat=False: output = mean(heads) ∈ R^{node_out}       → each head outputs node_out
        if concat:
            if node_out % num_heads != 0:
                raise ValueError(
                    f"node_out ({node_out}) must be divisible by num_heads ({num_heads}) "
                    f"when concat=True"
                )
            self.head_dim = node_out // num_heads
        else:
            self.head_dim = node_out  # each head → node_out; averaged → node_out

        self.out_dim = num_heads * self.head_dim  # total dim after msg_proj

        # Message transform: W · [h_u ; x_{u→v}]  (per head)
        self.msg_proj = nn.Linear(node_in + edge_dim, self.out_dim, bias=False)

        # Attention vector: a^T · [h_u ; h_v ; x_{u→v}]  (per head, scalar output)
        self.att_proj = nn.Linear(node_in + node_in + edge_dim, num_heads, bias=False)

        final_dim = num_heads * self.head_dim if concat else self.head_dim
        self.bias = nn.Parameter(torch.zeros(final_dim))
        self.norm = nn.LayerNorm(final_dim)

        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.msg_proj.weight)
        nn.init.xavier_uniform_(self.att_proj.weight)

    def forward(
        self,
        x: torch.Tensor,          # (N, node_in)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:
        """
        Returns:
            (N, node_out) — concatenated or averaged multi-head output
        """
        N = x.size(0)
        src, dst = edge_index  # src = u (source), dst = v (target)

        # --- messages: W · [h_u ; x_{u→v}] ---
        msg_input = torch.cat([x[src], edge_attr], dim=-1)      # (E, node_in + edge_dim)
        msgs = self.msg_proj(msg_input)                          # (E, H*head_dim)
        msgs = msgs.view(-1, self.num_heads, self.head_dim)      # (E, H, head_dim)

        # --- attention coefficients ---
        att_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)  # (E, 2*node_in + edge_dim)
        att_logits = self.att_proj(att_input)                        # (E, H)
        att_logits = F.leaky_relu(att_logits, self.negative_slope)

        # Softmax over incoming edges per destination node
        att = self._sparse_softmax(att_logits, dst, N)           # (E, H)
        att = self.dropout(att)

        # --- aggregate ---
        weighted = msgs * att.unsqueeze(-1)                      # (E, H, head_dim)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(weighted), weighted)

        if self.concat:
            out = out.reshape(N, self.num_heads * self.head_dim)
        else:
            out = out.mean(dim=1)                                # (N, head_dim)

        out = out + self.bias
        out = self.norm(out)
        return out

    @staticmethod
    def _sparse_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Per-node softmax over incoming edge logits. logits: (E, H)."""
        # Numerical stability: subtract max per destination node
        max_per_node = torch.full(
            (num_nodes, logits.size(1)), float('-inf'), device=logits.device, dtype=logits.dtype
        )
        max_per_node.scatter_reduce_(
            0, index.unsqueeze(-1).expand_as(logits), logits, reduce='amax', include_self=True
        )
        logits_shifted = logits - max_per_node[index]
        exp_logits = torch.exp(logits_shifted)

        sum_per_node = torch.zeros(num_nodes, logits.size(1), device=logits.device, dtype=logits.dtype)
        sum_per_node.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_logits), exp_logits)

        return exp_logits / (sum_per_node[index] + 1e-9)


# ===========================================================================
# GCN layer (kept for ablation)
# ===========================================================================

class GCNConvLayer(nn.Module):
    """Graph Convolutional Network layer (symmetric normalisation)."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = x.size(0)
        src, dst = edge_index

        # Add self-loops
        self_idx = torch.arange(N, device=x.device)
        ei = torch.cat([edge_index, self_idx.unsqueeze(0).expand(2, -1)], dim=1)

        # Degree normalisation
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, ei[1], torch.ones(ei.size(1), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e6)

        x = self.lin(self.dropout(x))
        norm = deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]
        msgs = x[ei[0]] * norm.unsqueeze(-1)

        out = torch.zeros_like(x)
        out.scatter_add_(0, ei[1].unsqueeze(-1).expand_as(msgs), msgs)
        return out


# ===========================================================================
# GATConvLayer  — thin alias kept for backward compat
# ===========================================================================

class GATConvLayer(EdgeAwareGATLayer):
    """Backward-compatible alias for EdgeAwareGATLayer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        edge_dim: Optional[int] = None,
    ):
        super().__init__(
            node_in=in_features,
            node_out=out_features,
            edge_dim=edge_dim if edge_dim else 1,
            num_heads=num_heads,
            dropout=dropout,
            concat=concat,
        )
        self._compat_edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            # fallback: dummy zero edge features
            E = edge_index.size(1)
            edge_attr = torch.zeros(E, self._compat_edge_dim or 1, device=x.device, dtype=x.dtype)
        return super().forward(x, edge_index, edge_attr)


# ===========================================================================
# §5.1  GNNBase — multi-layer wrapper
# ===========================================================================

class GNNBase(nn.Module):
    """
    Multi-layer GNN for node representation learning.

    Supports:
      - 'gat'       : EdgeAwareGATLayer  (default, matches §5.1)
      - 'gcn'       : GCNConvLayer
      - 'graphsage' : mean-aggregation SAGEConv (simplified)
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
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout_p = dropout
        self._edge_dim = edge_dim or 1

        if gnn_type not in ("gat", "gcn", "graphsage"):
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_f = input_dim if i == 0 else hidden_dim
            # last layer: no concat → output_dim; intermediate: concat → hidden_dim
            is_last = (i == num_layers - 1)
            out_f = output_dim if is_last else hidden_dim
            concat = not is_last

            if gnn_type == "gat":
                self.layers.append(
                    EdgeAwareGATLayer(
                        node_in=in_f,
                        node_out=out_f,
                        edge_dim=self._edge_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        concat=concat,
                    )
                )
            elif gnn_type == "gcn":
                self.layers.append(GCNConvLayer(in_f, out_f, dropout=dropout))
            elif gnn_type == "graphsage":
                self.layers.append(_SAGEConvLayer(in_f, out_f, dropout=dropout))

    def forward(
        self,
        x: torch.Tensor,           # (N, input_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: Optional[torch.Tensor] = None,  # (E, edge_dim)
    ) -> torch.Tensor:
        """
        Returns:
            (N, output_dim)
        """
        if edge_attr is None:
            E = edge_index.size(1)
            edge_attr = torch.zeros(E, self._edge_dim, device=x.device, dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            is_last = (i == self.num_layers - 1)
            if self.gnn_type == "gat":
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)

            if not is_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x


class _SAGEConvLayer(nn.Module):
    """Mean-aggregation GraphSAGE layer."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **_) -> torch.Tensor:
        N = x.size(0)
        src, dst = edge_index

        agg = torch.zeros_like(x)
        count = torch.zeros(N, 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), x[src])
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(src.size(0), 1, device=x.device))

        neigh_mean = agg / (count + 1e-9)
        out = self.lin_self(self.dropout(x)) + self.lin_neigh(neigh_mean)
        return self.norm(out)
