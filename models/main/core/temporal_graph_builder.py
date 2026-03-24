"""
Temporal Graph Builder: Construct transaction graphs within time windows

Partitions transactions into temporal windows and builds directed multigraphs
where nodes are addresses and edges are transactions.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import pandas as pd


class TemporalWindow:
    """Represents a single time window with its transaction graph."""
    
    def __init__(
        self,
        window_id: int,
        start_block: int,
        end_block: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ):
        self.window_id = window_id
        self.start_block = start_block
        self.end_block = end_block
        self.start_time = start_time
        self.end_time = end_time
        
        # Graph structure
        self.nodes = set()  # Set of addresses (nodes)
        self.edges = []  # List of edge dicts
        self.edge_index = []  # List of (from_idx, to_idx) tuples
        self.edge_features = []  # List of feature vectors
        self.edge_labels = []  # List of label
        
        # Address to node index mapping
        self.addr_to_idx = {}
        self.idx_to_addr = {}
        self.next_idx = 0
        
        # NetworkX graph for analysis
        self.graph = None
    
    def add_node(self, address: str) -> int:
        """
        Add node (address) to graph.
        
        Returns:
            Node index
        """
        addr_lower = address.lower()
        if addr_lower not in self.addr_to_idx:
            idx = self.next_idx
            self.addr_to_idx[addr_lower] = idx
            self.idx_to_addr[idx] = addr_lower
            self.nodes.add(addr_lower)
            self.next_idx += 1
            return idx
        return self.addr_to_idx[addr_lower]
    
    def add_edge(
        self,
        from_addr: str,
        to_addr: str,
        edge_features: np.ndarray,
        edge_label: int,
        tx_hash: str
    ) -> None:
        """
        Add edge (transaction) to graph.
        """
        from_idx = self.add_node(from_addr)
        to_idx = self.add_node(to_addr)
        
        self.edge_index.append((from_idx, to_idx))
        self.edge_features.append(edge_features)
        self.edge_labels.append(edge_label)
        
        self.edges.append({
            'from': from_addr,
            'to': to_addr,
            'tx_hash': tx_hash,
            'features': edge_features.copy()
        })
    
    def get_graph_data(self) -> Dict[str, torch.Tensor]:
        """
        Convert to PyTorch Geometric format.
        
        Returns:
            Dictionary with:
            - edge_index: (2, num_edges) tensor
            - edge_attr: (num_edges, feature_dim) tensor
            - edge_labels: (num_edges,) tensor
            - num_nodes: scalar
        """
        num_nodes = len(self.nodes)
        num_edges = len(self.edge_index)
        
        if num_edges == 0:
            # Empty graph
            return {
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'edge_attr': torch.zeros((0, 0), dtype=torch.float32),
                'edge_labels': torch.zeros(0, dtype=torch.long),
                'num_nodes': num_nodes,
                'addr_to_idx': self.addr_to_idx,
                'idx_to_addr': self.idx_to_addr
            }
        
        # Stack edge indices
        edge_idx_array = np.array(self.edge_index, dtype=np.int64)
        edge_index = torch.from_numpy(edge_idx_array.T)  # (2, num_edges)
        
        # Stack edge features
        edge_attr = torch.from_numpy(np.array(self.edge_features, dtype=np.float32))
        # (num_edges, feature_dim)
        
        # Edge labels
        edge_labels = torch.from_numpy(np.array(self.edge_labels, dtype=np.int64))
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_labels': edge_labels,
            'num_nodes': num_nodes,
            'addr_to_idx': self.addr_to_idx,
            'idx_to_addr': self.idx_to_addr
        }
    
    def build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph for analysis."""
        G = nx.DiGraph()
        
        # Add nodes
        G.add_nodes_from(self.nodes)
        
        # Add edges
        for edge_dict in self.edges:
            G.add_edge(edge_dict['from'], edge_dict['to'], tx_hash=edge_dict['tx_hash'])
        
        self.graph = G
        return G
    
    def get_node_stats(self) -> Dict[str, int]:
        """Get basic graph statistics."""
        if self.graph is None:
            self.build_networkx_graph()
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_multi_edges': len([e for e in self.edges if (e['from'], e['to']) in [(edge[0], edge[1]) for edge in self.edges]]),
            'density': nx.density(self.graph) if len(self.nodes) > 1 else 0.0
        }


class TemporalGraphBuilder:
    """
    Build temporal transaction graphs partitioned into time windows.
    
    Time window scheme:
    - Partition by block number (not timestamp for consistency)
    - Each window contains transactions in [start_block, end_block)
    - Δ = 1000 blocks per window
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        start_block: int = 4000000,
        end_block: int = 4010000
    ):
        """
        Args:
            window_size: Number of blocks per temporal window
            start_block: Starting block number
            end_block: Ending block number (exclusive)
        """
        self.window_size = window_size
        self.start_block = start_block
        self.end_block = end_block
        
        # Compute windows
        self.windows = []
        self._create_windows()
    
    def _create_windows(self) -> None:
        """Create temporal windows."""
        window_id = 0
        for start in range(self.start_block, self.end_block, self.window_size):
            end = min(start + self.window_size, self.end_block)
            window = TemporalWindow(
                window_id=window_id,
                start_block=start,
                end_block=end
            )
            self.windows.append(window)
            window_id += 1
    
    def get_window_for_block(self, block_number: int) -> Optional[TemporalWindow]:
        """Get temporal window containing given block."""
        for window in self.windows:
            if window.start_block <= block_number < window.end_block:
                return window
        return None
    
    def add_transaction(
        self,
        block_number: int,
        from_address: str,
        to_address: str,
        edge_features: np.ndarray,
        edge_label: int,
        tx_hash: str
    ) -> bool:
        """
        Add transaction (edge) to appropriate window.
        
        Returns:
            True if added successfully, False otherwise
        """
        window = self.get_window_for_block(block_number)
        if window is None:
            return False
        
        window.add_edge(from_address, to_address, edge_features, edge_label, tx_hash)
        return True
    
    def build_windows(self) -> List[Dict[str, torch.Tensor]]:
        """
        Convert all windows to PyTorch format.
        
        Returns:
            List of graph data dicts (one per window)
        """
        graph_data_list = []
        for window in self.windows:
            graph_data = window.get_graph_data()
            graph_data['window_id'] = window.window_id
            graph_data['start_block'] = window.start_block
            graph_data['end_block'] = window.end_block
            graph_data_list.append(graph_data)
        
        return graph_data_list
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Get statistics for all temporal windows.
        
        Returns:
            DataFrame with per-window statistics
        """
        stats_list = []
        for window in self.windows:
            stats = window.get_node_stats()
            stats['window_id'] = window.window_id
            stats['start_block'] = window.start_block
            stats['end_block'] = window.end_block
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def print_summary(self) -> None:
        """Print summary of temporal graph construction."""
        stats_df = self.get_statistics()
        
        print(f"Temporal Graph Builder Summary")
        print(f"=" * 60)
        print(f"Total windows: {len(self.windows)}")
        print(f"Block range: [{self.start_block}, {self.end_block})")
        print(f"Window size: {self.window_size} blocks")
        print(f"\nPer-window statistics:")
        print(stats_df.to_string(index=False))
        print(f"\nAggregate statistics:")
        print(f"  Total nodes: {stats_df['num_nodes'].sum()}")
        print(f"  Total edges: {stats_df['num_edges'].sum()}")
        print(f"  Avg edges/window: {stats_df['num_edges'].mean():.1f}")
        print(f"  Max edges/window: {stats_df['num_edges'].max()}")


class MultiWindowGraphDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for temporal graph windows.
    
    Each sample is a complete temporal graph.
    """
    
    def __init__(
        self,
        graph_data_list: List[Dict[str, torch.Tensor]],
        edge_feature_dim: int = 135
    ):
        """
        Args:
            graph_data_list: List of graph data dicts from TemporalGraphBuilder
            edge_feature_dim: Dimension of edge features
        """
        self.graph_data_list = graph_data_list
        self.edge_feature_dim = edge_feature_dim
    
    def __len__(self) -> int:
        return len(self.graph_data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
            - edge_index: (2, num_edges)
            - edge_attr: (num_edges, feature_dim)
            - edge_label: (num_edges,)
            - num_nodes: int
            - window_id: int
        """
        data = self.graph_data_list[idx]
        
        # Ensure edge_attr has correct dimension
        edge_attr = data.get('edge_attr', torch.zeros((0, self.edge_feature_dim)))
        if edge_attr.size(1) < self.edge_feature_dim:
            # Pad if necessary
            padding = torch.zeros(edge_attr.size(0), self.edge_feature_dim - edge_attr.size(1))
            edge_attr = torch.cat([edge_attr, padding], dim=1)
        elif edge_attr.size(1) > self.edge_feature_dim:
            # Truncate if necessary
            edge_attr = edge_attr[:, :self.edge_feature_dim]
        
        return {
            'edge_index': data['edge_index'],
            'edge_attr': edge_attr,
            'edge_label': data['edge_labels'],
            'num_nodes': data['num_nodes'],
            'window_id': data.get('window_id', -1),
            'addr_to_idx': data.get('addr_to_idx', {}),
            'idx_to_addr': data.get('idx_to_addr', {})
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for mini-batch.
        
        Combines multiple temporal graphs in batch.
        """
        batch_edge_index = []
        batch_edge_attr = []
        batch_edge_label = []
        
        node_offset = 0
        
        for graph in batch:
            edge_index = graph['edge_index']
            
            # Offset node indices
            if edge_index.size(1) > 0:
                edge_index = edge_index + node_offset
                batch_edge_index.append(edge_index)
                batch_edge_attr.append(graph['edge_attr'])
                batch_edge_label.append(graph['edge_label'])
            
            node_offset += graph['num_nodes']
        
        # Concatenate
        if batch_edge_index:
            edge_index = torch.cat(batch_edge_index, dim=1)
            edge_attr = torch.cat(batch_edge_attr, dim=0)
            edge_label = torch.cat(batch_edge_label, dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_feature_dim), dtype=torch.float32)
            edge_label = torch.zeros(0, dtype=torch.long)
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_label': edge_label,
            'num_nodes': node_offset,
            'batch_size': len(batch)
        }
