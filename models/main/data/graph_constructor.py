"""
Graph Constructor: Convert transactions to temporal graphs for GNN training.

Implements the paper's Transaction Graph Construction (§2-3):
- Temporal partitioning
- Node collection: V_k = {addresses in window T_k}
- Edge construction: one edge per external transaction
- Call trace processing and encoding

Handles:
- Loading transactions from PostgreSQL
- Building temporal graphs with proper graph structure
- DFS linearization and tokenization of call traces
- Creating PyTorch DataLoaders for batch training
- Caching for efficiency
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from collections import defaultdict
import pickle
from tqdm import tqdm
import networkx as nx

from .data_loader import TransactionDataLoader
from core import (
    EdgeFeatureExtractor,
    CallEventEmbedding
)


def linearize_call_trace(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Linearize call tree to sequence using DFS.
    
    Implements §4.2.1: Call Trace线性化表示
    
    Args:
        trace_data: Nested call trace dict from database
        
    Returns:
        List of call events in DFS order
    """
    calls = []
    
    def dfs(call: Dict, depth: int = 0):
        """DFS traversal of call tree"""
        calls.append({
            'call_type': call.get('type', 'CALL'),  # CALL, DELEGATECALL, STATICCALL, CREATE
            'to': call.get('to', '0x0'),  # Called contract address
            'input': call.get('input', '0x'),  # 4-byte selector + params
            'output': call.get('output', '0x'),
            'depth': depth,
            'revert': call.get('revert', False) or call.get('error', None) is not None,
            'gas': call.get('gas', 0),
            'gas_used': call.get('gasUsed', 0),
        })
        
        # DFS into subcalls
        for subcall in call.get('calls', []):
            dfs(subcall, depth + 1)
    
    if isinstance(trace_data, dict) and 'calls' in trace_data:
        for call in trace_data['calls']:
            dfs(call, 0)
    
    return calls

def _parse_value(val: Any) -> float:
    """Parse value which could be int, str, or hex string."""
    if val is None or val == 0:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('0x') or val.startswith('0X'):
            try:
                return float(int(val, 16))
            except (ValueError, TypeError):
                return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0

def tokenize_call_events(call_events: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize call events according to §4.2.2.
    
    Each call event c_i encoded as: x_i = e_type + e_contract + e_func + e_depth + e_exec
    
    Returns:
        Tuple of token ID arrays:
        - call_type_ids: (seq_len,)
        - contract_ids: extracted from 'to' address
        - func_ids: extracted from input selector
        - depth_ids: depth tokenized
        - exec_properties: (seq_len, 4) - [revert, gas_ratio, input_len, output_len]
    """
    seq_len = len(call_events)
    
    # Initialize token arrays
    call_type_ids = np.zeros(seq_len, dtype=np.int64)
    contract_ids = np.zeros(seq_len, dtype=np.int64)  
    func_ids = np.zeros(seq_len, dtype=np.int64)
    depth_ids = np.zeros(seq_len, dtype=np.int64)
    exec_properties = np.zeros((seq_len, 4), dtype=np.float32)
    
    # Call type mapping
    call_type_map = {
        'CALL': 1,
        'DELEGATECALL': 2,
        'STATICCALL': 3,
        'CREATE': 4,
        'CREATE2': 5
    }
    
    for i, event in enumerate(call_events):
        # Token 1: Call type
        call_type_ids[i] = call_type_map.get(event['call_type'], 0)
        
        # Token 2: Contract address (hash to index)
        to_addr = event['to'].lower()
        contract_ids[i] = hash(to_addr) % 10000  # Fixed vocabulary size
        
        # Token 3: Function selector (first 4 bytes of input)
        input_str = event['input']
        if len(input_str) >= 10:  # 0x + 8 hex chars
            func_selector = int(input_str[2:10], 16)
        else:
            func_selector = 0
        func_ids[i] = func_selector % 10000
        
        # Token 4: Call depth
        depth_ids[i] = min(event['depth'], 15)  # Cap depth at 15
        
        # Token 5: Execution properties [revert_flag, gas_used_ratio, input_len, output_len]
        input_len = (len(event['input']) - 2) // 2  # Bytes
        output_len = (len(event['output']) - 2) // 2
        gas_used = _parse_value(event.get('gas_used', 0))
        gas = _parse_value(event.get('gas', 0))
        gas_ratio = min(gas_used / max(gas, 1), 1.0)
        
        exec_properties[i] = [
            float(event['revert']),
            gas_ratio,
            min(input_len / 1000.0, 1.0),  # Normalize to [0,1]
            min(output_len / 1000.0, 1.0)
        ]
    
    return call_type_ids, contract_ids, func_ids, depth_ids, exec_properties


class TemporalGraphWithTraces:
    """
    Represents a single temporal window graph with traces.
    
    Implements §2-3: Transaction Graph Construction
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed multigraph for temporal window
        self.edges = []  # List of (from_addr, to_addr, tx_data)
        self.nodes_set: Set[str] = set()  # V_k
        self.edge_features = []  # x_e for each edge
        self.edge_labels = []  # y_e for each edge
        self.node_to_idx: Dict[str, int] = {}  # Mapping addresses to node indices
    
    def add_transaction(
        self,
        tx_hash: str,
        from_addr: str,
        to_addr: str,
        value: int,
        gas_used: int,
        trace_data: Dict[str, Any],
        is_suspicious: int,
        external_feat_dim: int = 6
    ) -> None:
        """
        Add a transaction as an edge to the temporal graph.
        
        Args:
            tx_hash: Transaction hash
            from_addr, to_addr: Source and destination addresses
            value, gas_used: Transaction properties
            trace_data: Complete call trace
            is_suspicious: Label
            external_feat_dim: Dimension of external features
        """
        # Add nodes to V_k
        self.nodes_set.add(from_addr)
        self.nodes_set.add(to_addr)
        
        # Add edge to graph
        self.edges.append({
            'from': from_addr,
            'to': to_addr,
            'hash': tx_hash,
            'value': value,
            'gas_used': gas_used
        })
        
        # Extract external features (§4.1)
        external_features = np.array([
            float(value),
            float(gas_used),
            0.0,  # calldata_length - computed later if needed
            1.0 if to_addr != '0x' * 40 else 0.0,  # is_contract_call
            0.0,  # is_revert (from trace)
            0.0   # nonce_position (from context)
        ], dtype=np.float32) / np.array([1e18, 1e9, 1e6, 1.0, 1.0, 1.0])  # Normalization
        
        # Process internal call trace (§4.2)
        call_events = linearize_call_trace(trace_data)
        call_type_ids, contract_ids, func_ids, depth_ids, exec_properties = \
            tokenize_call_events(call_events)
        
        # Create trace encoding inputs
        trace_features = {
            'call_type_ids': call_type_ids,
            'contract_ids': contract_ids,
            'func_ids': func_ids,
            'depth_ids': depth_ids,
            'exec_properties': exec_properties,
            'sequence_length': len(call_events)
        }
        
        # Store combined edge features: x_e = [x_e^external ; x_e^internal]
        edge_feature = {
            'tx_hash': tx_hash,
            'external_features': external_features,
            'trace_features': trace_features,
            'from_addr': from_addr,
            'to_addr': to_addr
        }
        
        self.edge_features.append(edge_feature)
        self.edge_labels.append(is_suspicious)
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize graph structure and convert to PyTorch format.
        
        Returns:
            Dictionary with PyTorch tensors ready for GNN
        """
        # Create node index mapping V_k → {0, 1, ..., |V_k|-1}
        sorted_nodes = sorted(self.nodes_set)
        self.node_to_idx = {addr: idx for idx, addr in enumerate(sorted_nodes)}
        num_nodes = len(sorted_nodes)
        
        # Convert edges to edge_index format (2, num_edges)
        edge_index = []
        for i, edge in enumerate(self.edges):
            from_idx = self.node_to_idx[edge['from']]
            to_idx = self.node_to_idx[edge['to']]
            edge_index.append([from_idx, to_idx])
        
        if edge_index:
            edge_index = np.array(edge_index).T  # (2, num_edges)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        # Create node features (initialized as zeros + will be updated by GNN)
        node_features = np.zeros((num_nodes, 7), dtype=np.float32)  # Placeholder
        
        return {
            'num_nodes': num_nodes,
            'num_edges': len(self.edges),
            'edge_index': edge_index,
            'edge_features': self.edge_features,
            'edge_labels': np.array(self.edge_labels, dtype=np.int64),
            'node_features': node_features,
            'node_to_idx': self.node_to_idx,
            'nodes': sorted_nodes
        }


class GraphConstructor:
    """Build temporal graphs from transaction data following the paper's algorithm."""
    
    def __init__(
        self,
        data_loader: TransactionDataLoader,
        feature_extractor: EdgeFeatureExtractor,
        temporal_window: int = 1000,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_loader: TransactionDataLoader instance
            feature_extractor: EdgeFeatureExtractor instance
            temporal_window: Block window size (§3: Δ)
            cache_dir: Directory for caching graphs
        """
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor
        self.temporal_window = temporal_window
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, start_block: int, end_block: int) -> Optional[Path]:
        """Get cache file path."""
        if not self.cache_dir:
            return None
        name = f"graphs_{start_block}_{end_block}"
        return self.cache_dir / f"{name}.pkl"
    
    def load_transactions(
        self,
        start_block: int,
        end_block: int,
        chunk_size: int = 5000
    ) -> List[Dict[str, Any]]:
        """Load all transactions from database."""
        transactions = []
        
        iterator = self.data_loader.stream_transactions(
            start_block=start_block,
            end_block=end_block,
            chunk_size=chunk_size
        )
        
        for chunk in iterator:
            for idx, row in chunk.iterrows():
                tx = {
                    'block_number': int(row['block_number']),
                    'transaction_hash': str(row['transaction_hash']),
                    'transaction_index': int(row['transaction_index']),
                    'from_address': str(row['from_address']).lower(),
                    'to_address': str(row['to_address']).lower(),
                    'value': int(row['value']),
                    'gas_used': int(row['gas_used']),
                    'logs': row['logs'],
                    'trace_data': row['trace_data'],
                    'timestamp': row['timestamp'].timestamp(),
                    'is_suspicious': int(row['is_suspicious'])
                }
                transactions.append(tx)
        
        return transactions
    
    def build_temporal_graphs(
        self,
        transactions: List[Dict[str, Any]],
        start_block: int,
        end_block: int
    ) -> List[Dict[str, Any]]:
        """
        Build temporal graphs following §2-3.
        
        For each time window T_k:
        1. Collect V_k = {addresses in window}
        2. Build E_k = {edges (u→v) for each external transaction}
        3. Combine external + internal features into x_e
        """
        # Group transactions by temporal window (§3)
        windows = defaultdict(list)
        for tx in transactions:
            window_id = (tx['block_number'] - start_block) // self.temporal_window
            windows[window_id].append(tx)
        
        # Build graph for each window
        graphs = []
        
        for window_id in sorted(windows.keys()):
            window_txs = windows[window_id]
            if not window_txs:
                continue
            
            # Create temporal graph for this window
            temporal_graph = TemporalGraphWithTraces()
            
            # Add all transactions as edges
            for tx in window_txs:
                temporal_graph.add_transaction(
                    tx_hash=tx['transaction_hash'],
                    from_addr=tx['from_address'],
                    to_addr=tx['to_address'],
                    value=tx['value'],
                    gas_used=tx['gas_used'],
                    trace_data=tx['trace_data'],
                    is_suspicious=tx['is_suspicious']
                )
            
            # Finalize and convert to PyTorch format
            graph_data = temporal_graph.finalize()
            graph_data['window_id'] = window_id
            graph_data['block_start'] = min(tx['block_number'] for tx in window_txs)
            graph_data['block_end'] = max(tx['block_number'] for tx in window_txs)
            
            graphs.append(graph_data)
        
        return graphs
    
    def construct_graphs(
        self,
        start_block: int,
        end_block: int,
        use_cache: bool = True,
        force_rebuild: bool = False
    ) -> List[Dict[str, Any]]:
        """Main method to construct temporal graphs."""
        # Check cache
        if use_cache and not force_rebuild:
            cache_path = self.get_cache_path(start_block, end_block)
            if cache_path and cache_path.exists():
                print(f"Loading cached graphs from {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        print(f"Loading transactions from block {start_block} to {end_block}...")
        transactions = self.load_transactions(start_block, end_block)
        print(f"Loaded {len(transactions)} transactions")
        
        print("Building temporal graphs...")
        graphs = self.build_temporal_graphs(transactions, start_block, end_block)
        print(f"Built {len(graphs)} temporal graphs")
        
        # Save to cache
        if use_cache:
            cache_path = self.get_cache_path(start_block, end_block)
            if cache_path:
                print(f"Saving graphs to cache: {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(graphs, f)
        
        return graphs


class GraphDataLoader:
    """Create PyTorch DataLoaders from temporal graphs."""
    
    def __init__(
        self,
        graphs: List[Dict[str, Any]],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Args:
            graphs: List of temporal window graphs
            batch_size: Batch size for DataLoader
            shuffle: Shuffle batches
            num_workers: Number of workers for data loading
        """
        self.graphs = graphs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def create_dataloaders(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        mode: str = 'edge'
    ) -> Tuple:
        """
        Create train/val/test DataLoaders.
        
        Args:
            train_ratio, val_ratio, test_ratio: Data split ratios
            mode: 'edge'  → EdgeLevelDataset (batch_size edges per step, for non-GNN models)
                  'graph' → GraphWindowDataset (1 full graph window per step, for GNN models)
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        num_graphs = len(self.graphs)
        train_size = int(num_graphs * train_ratio)
        val_size = int(num_graphs * val_ratio)
        test_size = num_graphs - train_size - val_size

        # Determine graph-level binary labels for stratification: has any suspicious edge
        graph_has_suspicious = np.array([
            1 if np.any(np.array(graph.get('edge_labels', [])) == 1) else 0
            for graph in self.graphs
        ], dtype=np.int32)

        normal_indices = np.where(graph_has_suspicious == 0)[0]
        suspicious_indices = np.where(graph_has_suspicious == 1)[0]

        # Shuffle groups (if requested)
        if self.shuffle:
            np.random.shuffle(normal_indices)
            np.random.shuffle(suspicious_indices)

        # Build split with at least one suspicious in each split when possible
        train_indices = []
        val_indices = []
        test_indices = []

        if len(suspicious_indices) >= 3:
            train_indices.append(suspicious_indices[0])
            val_indices.append(suspicious_indices[1])
            test_indices.append(suspicious_indices[2])
            remaining_suspicious = suspicious_indices[3:]
        else:
            # If insufficient suspicious graphs for all splits, keep all in train and continue
            train_indices.extend(suspicious_indices)
            remaining_suspicious = np.array([], dtype=int)

        # Distribute remaining suspicious according to ratio
        if len(remaining_suspicious) > 0:
            total_alloc = train_size + val_size + test_size - len(train_indices) - len(val_indices) - len(test_indices)
            if total_alloc <= 0:
                remaining_suspicious = np.array([], dtype=int)
            else:
                # Use proportional allocation by split sizes
                splits = [train_size, val_size, test_size]
                cap = [max(0, s - len(a)) for s, a in zip(splits, [train_indices, val_indices, test_indices])]
                for idx in remaining_suspicious:
                    # choose split with highest remaining proportion
                    dist = [cap[0] / max(1, train_size), cap[1] / max(1, val_size), cap[2] / max(1, test_size)]
                    target = int(np.argmax(dist))
                    if target == 0 and len(train_indices) < train_size:
                        train_indices.append(idx)
                        cap[0] -= 1
                    elif target == 1 and len(val_indices) < val_size:
                        val_indices.append(idx)
                        cap[1] -= 1
                    elif target == 2 and len(test_indices) < test_size:
                        test_indices.append(idx)
                        cap[2] -= 1
                    else:
                        # fallback append where has space
                        if len(train_indices) < train_size:
                            train_indices.append(idx)
                        elif len(val_indices) < val_size:
                            val_indices.append(idx)
                        elif len(test_indices) < test_size:
                            test_indices.append(idx)

        # Fill each split up to desired size with normal graphs
        def fill_split(split_list, target_size):
            while len(split_list) < target_size and len(normal_indices) > 0:
                split_list.append(normal_indices[0])
                normal_indices = normal_indices[1:]
            return split_list

        # Python closure does not allow rebinding outer normal_indices directly; use new variable
        normal_remaining = list(normal_indices)
        def fill(split_list, target_size):
            nonlocal normal_remaining
            while len(split_list) < target_size and normal_remaining:
                split_list.append(normal_remaining.pop(0))
            return split_list

        train_indices = fill(train_indices, train_size)
        val_indices = fill(val_indices, val_size)
        test_indices = fill(test_indices, test_size)

        # Add any leftovers to train/test if something remains
        leftovers = normal_remaining
        for idx in leftovers:
            if len(train_indices) < train_size:
                train_indices.append(idx)
            elif len(val_indices) < val_size:
                val_indices.append(idx)
            elif len(test_indices) < test_size:
                test_indices.append(idx)
            else:
                break

        # Final check
        assert len(train_indices) + len(val_indices) + len(test_indices) == num_graphs, \
            f"Split mismatch: {len(train_indices)}, {len(val_indices)}, {len(test_indices)} vs {num_graphs}"

        # Convert to numpy arrays
        train_indices = np.array(train_indices, dtype=int)
        val_indices = np.array(val_indices, dtype=int)
        test_indices = np.array(test_indices, dtype=int)
        
        # Create datasets
        train_graphs = [self.graphs[i] for i in train_indices]
        val_graphs = [self.graphs[i] for i in val_indices]
        test_graphs = [self.graphs[i] for i in test_indices]
        
        if mode == 'graph':
            # GNN models: 1 full temporal window per step (provides edge_index + node_features)
            train_loader = torch.utils.data.DataLoader(
                GraphWindowDataset(train_graphs),
                batch_size=1,
                shuffle=self.shuffle,
                collate_fn=_collate_graph_window,
                num_workers=self.num_workers
            )
            val_loader = torch.utils.data.DataLoader(
                GraphWindowDataset(val_graphs),
                batch_size=1,
                shuffle=False,
                collate_fn=_collate_graph_window,
                num_workers=self.num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                GraphWindowDataset(test_graphs),
                batch_size=1,
                shuffle=False,
                collate_fn=_collate_graph_window,
                num_workers=self.num_workers
            )
        else:
            # Non-GNN models: edge-level batching to avoid GPU OOM
            train_loader = torch.utils.data.DataLoader(
                EdgeLevelDataset(train_graphs),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            val_loader = torch.utils.data.DataLoader(
                EdgeLevelDataset(val_graphs),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                EdgeLevelDataset(test_graphs),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        
        return train_loader, val_loader, test_loader


class GraphDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for temporal graphs following the paper's format."""
    
    def __init__(self, graphs: List[Dict[str, Any]]):
        """
        Args:
            graphs: List of temporal graph dicts with structure:
                - num_nodes, num_edges
                - edge_index, edge_features, edge_labels
                - node_features
        """
        self.graphs = graphs
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return complete graph data."""
        return self.graphs[idx]


class EdgeLevelDataset(torch.utils.data.Dataset):
    """
    Edge-level dataset: each sample is one transaction edge.

    Replaces graph-level batching so that `batch_size` controls how many
    *edges* (transactions) are processed per forward pass, preventing OOM
    when individual temporal graphs contain tens of thousands of edges.
    """

    MAX_SEQ_LEN = 256

    def __init__(self, graphs: List[Dict[str, Any]]):
        self.graphs = graphs
        # Build flat index (graph_idx, edge_idx_in_graph) without copying data
        self.index: List[Tuple[int, int]] = []
        for g_idx, graph in enumerate(graphs):
            for e_idx in range(graph['num_edges']):
                self.index.append((g_idx, e_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g_idx, e_idx = self.index[idx]
        graph = self.graphs[g_idx]
        edge_feat = graph['edge_features'][e_idx]
        label = int(graph['edge_labels'][e_idx])

        max_seq_len = self.MAX_SEQ_LEN
        ext_feat = np.asarray(edge_feat['external_features'], dtype=np.float32)
        trace_feat = edge_feat['trace_features']
        seq_len = trace_feat['sequence_length']
        max_len = min(seq_len, max_seq_len)

        call_type = np.zeros(max_seq_len, dtype=np.int64)
        contract  = np.zeros(max_seq_len, dtype=np.int64)
        func_id   = np.zeros(max_seq_len, dtype=np.int64)
        depth     = np.zeros(max_seq_len, dtype=np.int64)
        exec_prop = np.zeros((max_seq_len, 4), dtype=np.float32)
        mask      = np.zeros(max_seq_len, dtype=bool)

        if max_len > 0:
            call_type[:max_len] = trace_feat['call_type_ids'][:max_len]
            contract[:max_len]  = trace_feat['contract_ids'][:max_len]
            func_id[:max_len]   = trace_feat['func_ids'][:max_len]
            depth[:max_len]     = trace_feat['depth_ids'][:max_len]
            exec_prop[:max_len] = trace_feat['exec_properties'][:max_len]
            mask[:max_len]      = True

        return {
            'external_features':  torch.from_numpy(ext_feat),
            'call_type_ids':      torch.from_numpy(call_type),
            'contract_ids':       torch.from_numpy(contract),
            'func_selector_ids':  torch.from_numpy(func_id),
            'depths':             torch.from_numpy(depth),
            'exec_properties':    torch.from_numpy(exec_prop),
            'trace_mask':         torch.from_numpy(mask),
            'labels':             torch.tensor(label, dtype=torch.long),
            'num_edges':          torch.tensor(1, dtype=torch.long),
        }


def collate_graph_batch(
    batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Batch multiple temporal graphs into a single mini-batch.
    
    Implements proper graph batching with edge-level features.
    All edges in the batch are flattened into sequences.
    
    Args:
        batch: List of temporal graphs
    
    Returns:
        Batched data with model inputs (§4-5)
    """
    max_seq_len = 256  # Max trace sequence length
    
    # Collect all edges and their features from all graphs in batch
    all_external_features = []
    all_call_type_ids = []
    all_contract_ids = []
    all_func_selector_ids = []
    all_depths = []
    all_exec_properties = []
    all_trace_masks = []
    all_labels = []
    
    # For graph structure (optional - for future GNN layer)
    all_edge_indices = []
    node_offset = 0
    
    for graph in batch:
        # Each edge_features list contains one dict per edge
        num_edges = graph['num_edges']
        
        if num_edges == 0:
            continue
        
        # Process each edge
        for edge_feature_dict in graph['edge_features']:
            # External features (§4.1)
            ext_feat = edge_feature_dict['external_features']  # (7,)
            all_external_features.append(torch.from_numpy(ext_feat).float())
            
            # Internal trace features (§4.2)
            trace_feat = edge_feature_dict['trace_features']
            
            # Tokenized trace sequences
            seq_len = trace_feat['sequence_length']
            max_len = min(seq_len, max_seq_len)
            
            # Pad sequences to max_seq_len
            call_type = torch.zeros(max_seq_len, dtype=torch.long)
            contract = torch.zeros(max_seq_len, dtype=torch.long)
            func_id = torch.zeros(max_seq_len, dtype=torch.long)
            depth = torch.zeros(max_seq_len, dtype=torch.long)
            exec_prop = torch.zeros((max_seq_len, 4), dtype=torch.float32)
            
            # Fill in actual values
            call_type[:max_len] = torch.from_numpy(trace_feat['call_type_ids'][:max_len]).long()
            contract[:max_len] = torch.from_numpy(trace_feat['contract_ids'][:max_len]).long()
            func_id[:max_len] = torch.from_numpy(trace_feat['func_ids'][:max_len]).long()
            depth[:max_len] = torch.from_numpy(trace_feat['depth_ids'][:max_len]).long()
            exec_prop[:max_len] = torch.from_numpy(trace_feat['exec_properties'][:max_len]).float()
            
            # Attention mask: True for valid positions
            mask = torch.zeros(max_seq_len, dtype=torch.bool)
            mask[:max_len] = True
            
            all_call_type_ids.append(call_type)
            all_contract_ids.append(contract)
            all_func_selector_ids.append(func_id)
            all_depths.append(depth)
            all_exec_properties.append(exec_prop)
            all_trace_masks.append(mask)
        
        # Labels (one per edge)
        labels = torch.from_numpy(graph['edge_labels']).long()
        all_labels.append(labels)
        
        # Optional: Graph structure for GNN (offset nodes for batching)
        if graph['edge_index'].shape[1] > 0:  # Has edges
            edge_index = torch.from_numpy(graph['edge_index']).long()
            edge_index += node_offset
            all_edge_indices.append(edge_index)
        
        node_offset += graph['num_nodes']
    
    # Stack all features
    num_edges = len(all_labels)
    num_edges_total = sum(len(l) for l in all_labels)
    
    if num_edges_total == 0:
        # Empty batch - return zeros
        return {
            'external_features': torch.zeros((0, 7), dtype=torch.float32),
            'call_type_ids': torch.zeros((0, max_seq_len), dtype=torch.long),
            'contract_ids': torch.zeros((0, max_seq_len), dtype=torch.long),
            'func_selector_ids': torch.zeros((0, max_seq_len), dtype=torch.long),
            'depths': torch.zeros((0, max_seq_len), dtype=torch.long),
            'exec_properties': torch.zeros((0, max_seq_len, 4), dtype=torch.float32),
            'trace_mask': torch.zeros((0, max_seq_len), dtype=torch.bool),
            'labels': torch.zeros((0,), dtype=torch.long)
        }
    
    # Stack tensors
    return {
        'external_features': torch.stack(all_external_features),  # (num_edges, 7)
        'call_type_ids': torch.stack(all_call_type_ids),  # (num_edges, seq_len)
        'contract_ids': torch.stack(all_contract_ids),
        'func_selector_ids': torch.stack(all_func_selector_ids),
        'depths': torch.stack(all_depths),
        'exec_properties': torch.stack(all_exec_properties),  # (num_edges, seq_len, 4)
        'trace_mask': torch.stack(all_trace_masks),  # (num_edges, seq_len)
        'labels': torch.cat(all_labels),  # (num_edges,)
        'num_edges': num_edges_total
    }


# ============================================================================
# GraphWindowDataset — for GNN-based models (provides edge_index + node_features)
# ============================================================================

class GraphWindowDataset(torch.utils.data.Dataset):
    """
    Graph-window-level dataset: each sample is a complete temporal graph window.

    Used for GNN-based models (models 5-8) where message passing requires
    the full graph topology (edge_index, node_features).  The GNNWindowTrainer
    processes each window in a 2-stage loop:
      1. mini-batch trace encoding (avoids OOM)
      2. full-graph GNN → mini-batch edge classification
    """

    MAX_SEQ_LEN = 256

    def __init__(self, graphs: List[Dict[str, Any]]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the full graph window as pre-processed tensors."""
        graph = self.graphs[idx]
        num_edges = graph['num_edges']
        max_seq_len = self.MAX_SEQ_LEN

        # Pre-allocate edge-level tensors
        external_features = np.zeros((num_edges, 6), dtype=np.float32)
        call_type_ids     = np.zeros((num_edges, max_seq_len), dtype=np.int64)
        contract_ids      = np.zeros((num_edges, max_seq_len), dtype=np.int64)
        func_selector_ids = np.zeros((num_edges, max_seq_len), dtype=np.int64)
        depths            = np.zeros((num_edges, max_seq_len), dtype=np.int64)
        exec_properties   = np.zeros((num_edges, max_seq_len, 4), dtype=np.float32)
        trace_masks       = np.zeros((num_edges, max_seq_len), dtype=bool)

        for i, edge_feat in enumerate(graph['edge_features']):
            ext = np.asarray(edge_feat['external_features'], dtype=np.float32)
            external_features[i, :len(ext)] = ext[:6]  # clip to 6 dims

            tf = edge_feat['trace_features']
            seq_len = tf['sequence_length']
            max_len = min(seq_len, max_seq_len)
            if max_len > 0:
                call_type_ids[i, :max_len]       = tf['call_type_ids'][:max_len]
                contract_ids[i, :max_len]         = tf['contract_ids'][:max_len]
                func_selector_ids[i, :max_len]    = tf['func_ids'][:max_len]
                depths[i, :max_len]               = tf['depth_ids'][:max_len]
                exec_properties[i, :max_len]      = tf['exec_properties'][:max_len]
                trace_masks[i, :max_len]          = True

        edge_index_np = graph['edge_index']          # (2, E) numpy
        node_feats_np = graph.get('node_features')   # (N, D) numpy or None
        if node_feats_np is None:
            node_feats_np = np.zeros((graph['num_nodes'], 6), dtype=np.float32)

        return {
            # Edge-level features
            'external_features':  torch.from_numpy(external_features),   # (E, 6)
            'call_type_ids':      torch.from_numpy(call_type_ids),        # (E, 256)
            'contract_ids':       torch.from_numpy(contract_ids),
            'func_selector_ids':  torch.from_numpy(func_selector_ids),
            'depths':             torch.from_numpy(depths),
            'exec_properties':    torch.from_numpy(exec_properties),      # (E, 256, 4)
            'trace_mask':         torch.from_numpy(trace_masks),          # (E, 256)
            'labels':             torch.from_numpy(
                                      np.asarray(graph['edge_labels'], dtype=np.int64)
                                  ),                                       # (E,)
            # Graph structure
            'edge_index':         torch.from_numpy(
                                      np.asarray(edge_index_np, dtype=np.int64)
                                  ),                                       # (2, E)
            'node_features':      torch.from_numpy(
                                      np.asarray(node_feats_np, dtype=np.float32)
                                  ),                                       # (N, D)
            'num_edges':          torch.tensor(num_edges, dtype=torch.long),
        }


def _collate_graph_window(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for GraphWindowDataset (batch_size=1).
    Removes the outer list wrapper added by DataLoader.
    """
    assert len(batch) == 1, "GraphWindowDataset must be used with batch_size=1"
    return batch[0]
