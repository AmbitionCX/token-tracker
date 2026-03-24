"""
Edge Feature Extractor: Extract and combine external + internal call trace features

Processes raw transaction data and creates edge feature representations
combining external transaction properties with internal call trace information.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
import math

# Import from trace_encoder
from .trace_encoder import CallEventEmbedding


class ExternalTransactionFeatures:
    """
    Extract features from external transaction level.
    
    Features:
    - value: Whether ETH is transferred
    - gas_used: Actual gas consumed
    - calldata_length: Length of transaction input data
    - is_contract_call: Whether target address is a contract
    - is_revert: Whether transaction reverted
    - nonce_position: Relative nonce position (burst detection)
    """
    
    FEATURE_NAMES = [
        'value',
        'gas_used',
        'calldata_length',
        'is_contract_call',
        'is_revert',
        'nonce_position'
    ]
    
    def __init__(self):
        self.n_features = len(self.FEATURE_NAMES)
    
    @staticmethod
    def _normalize_value(value: float, max_val: float = 1e18) -> float:
        """Normalize numeric values using log scale."""
        if value <= 0:
            return 0.0
        return math.log(value + 1e-9) / math.log(max_val + 1e-9)
    
    def extract(self, tx: Dict[str, Any]) -> np.ndarray:
        """
        Extract external transaction features.
        
        Args:
            tx: Transaction record with fields:
                - value: ETH value transferred
                - gas_used: Gas consumed
                - trace_data: Call trace (for calldata_length)
                - to_address: Target address
                - is_contract_call: Flag
                - is_revert: Flag (if available)
        
        Returns:
            np.ndarray of shape (7,) with normalized features
        """
        features = []
        
        # 1. Value: normalize to [0, 1] using log scale
        value = float(tx.get('value', 0) or 0)
        value_norm = self._normalize_value(value)
        features.append(value_norm)
        
        # 2. Gas used: normalize
        gas_used = float(tx.get('gas_used', 0) or 0)
        gas_norm = self._normalize_value(gas_used, max_val=10e6)
        features.append(gas_norm)
        
        # 3. Calldata length: from trace_data
        calldata_length = 0
        trace_data = tx.get('trace_data')
        if isinstance(trace_data, dict) and 'input' in trace_data:
            input_hex = trace_data['input']
            if isinstance(input_hex, str) and input_hex.startswith('0x'):
                calldata_length = (len(input_hex) - 2) // 2
        calldata_norm = self._normalize_value(calldata_length, max_val=10000)
        features.append(calldata_norm)
        
        # 4. Is contract call: to_address is contract (heuristic or known list)
        is_contract = float(tx.get('is_contract_call', 0))
        features.append(is_contract)
        
        # 5. Is revert: check if transaction failed
        is_revert = 0.0
        if isinstance(trace_data, dict):
            # Check if trace output is "0x" (empty/failed)
            output = trace_data.get('output', '0x')
            if output == '0x' and 'calls' not in trace_data:
                is_revert = 1.0
        features.append(is_revert)
        
        # 6. Nonce position (burst behavior): placeholder
        # In production, would need sequence of transactions from same account
        nonce_position = tx.get('transaction_index', 0) / 100.0  # Normalize by typical block size
        nonce_position = min(nonce_position, 1.0)
        features.append(nonce_position)
        
        return np.array(features, dtype=np.float32)


class InternalCallTraceFeatures:
    """
    Extract features from internal call trace (call events).
    
    Processes the call trace tree and returns:
    - Linearized sequence of call events
    - Token IDs for embedding layer
    - Execution properties
    """
    
    # Call type mappings
    CALL_TYPES = {
        'CALL': 1,
        'DELEGATECALL': 2,
        'STATICCALL': 3,
        'CREATE': 4,
        'CREATE2': 5,
        'SELFDESTRUCT': 6,
    }
    
    def __init__(
        self,
        max_sequence_length: int = 256,
        contract_vocab_dir: Optional[str] = None
    ):
        self.max_sequence_length = max_sequence_length
        self.contract_vocab = {}  # Contract address → ID mapping (load from file if provided)
        self.next_contract_id = 1
    
    def _get_contract_id(self, address: str) -> int:
        """Get or assign ID for contract address."""
        addr_lower = address.lower()
        if addr_lower not in self.contract_vocab:
            self.contract_vocab[addr_lower] = self.next_contract_id
            self.next_contract_id += 1
        return self.contract_vocab[addr_lower]
    
    def _linearize_trace(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Linearize call trace tree using DFS pre-order traversal.
        
        Args:
            trace: Call trace dict with 'type', 'to', 'input', 'calls', etc.
        
        Returns:
            List of linearized call events
        """
        events = []
        depth = trace.get('_depth', 0)  # Track depth in recursion
        
        def dfs(node, current_depth=0):
            # Add current node as event
            event = {
                'type': node.get('type', 'CALL'),
                'to': node.get('to', '0x0'),
                'function_selector': self._extract_function_selector(node.get('input', '0x')),
                'depth': current_depth,
                'input_size': len(node.get('input', '0x')) // 2 - 1 if node.get('input', '0x').startswith('0x') else 0,
                'output_size': len(node.get('output', '0x')) // 2 - 1 if node.get('output', '0x').startswith('0x') else 0,
                'gas_used': int(node.get('gasUsed', '0x0'), 16) if isinstance(node.get('gasUsed'), str) else node.get('gasUsed', 0),
                'has_revert': 1 if node.get('output', '0x') == '0x' and len(node.get('calls', [])) == 0 else 0,
            }
            events.append(event)
            
            # Recurse on sub-calls
            for subcall in node.get('calls', []):
                dfs(subcall, current_depth + 1)
        
        dfs(trace)
        return events
    
    @staticmethod
    def _extract_function_selector(input_hex: str) -> str:
        """
        Extract 4-byte function selector from calldata.
        
        Args:
            input_hex: Input data in hex format (e.g., "0x1234abcd...")
        
        Returns:
            4-byte selector as hex string (e.g., "0x1234abcd")
        """
        if not isinstance(input_hex, str) or len(input_hex) < 10:
            return "0x00000000"
        return input_hex[:10]
    
    def extract(self, trace_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and prepare internal call trace features.
        
        Args:
            trace_data: Raw call trace dict from transaction
        
        Returns:
            Tuple of:
            - call_type_ids: (seq_len,) - token IDs for call types
            - contract_ids: (seq_len,) - token IDs for contracts
            - func_selector_ids: (seq_len,) - token IDs for function selectors
            - depths: (seq_len,) - call depths
            - exec_properties: (seq_len, 4) - execution properties [revert, input_size, output_size, gas_used]
        """
        if not isinstance(trace_data, dict):
            # Return empty arrays if no trace
            return self._empty_sequence()
        
        # Linearize trace
        events = self._linearize_trace(trace_data)
        
        if not events:
            return self._empty_sequence()
        
        # Limit to max sequence length
        events = events[:self.max_sequence_length]
        seq_len = len(events)
        
        # Extract token IDs and properties
        call_type_ids = np.zeros(seq_len, dtype=np.int32)
        contract_ids = np.zeros(seq_len, dtype=np.int32)
        func_selector_ids = np.zeros(seq_len, dtype=np.int32)
        depths = np.zeros(seq_len, dtype=np.int32)
        exec_properties = np.zeros((seq_len, 4), dtype=np.float32)
        
        # Function selector hash (simple hash to ID)
        func_selector_cache = {}
        next_func_id = 1
        
        for i, event in enumerate(events):
            # Call type ID
            call_type = self.CALL_TYPES.get(event['type'], 0)
            call_type_ids[i] = call_type
            
            # Contract ID (with hashing for vocab size management)
            contract_addr = event['to'].lower()
            contract_id = self._get_contract_id(contract_addr) % 50000  # Limit vocab
            contract_ids[i] = contract_id
            
            # Function selector ID
            func_sel = event['function_selector']
            if func_sel not in func_selector_cache:
                func_selector_cache[func_sel] = next_func_id
                next_func_id += 1
            func_selector_ids[i] = func_selector_cache[func_sel] % 100000
            
            # Depth
            depths[i] = min(event['depth'], 49)  # Limit to max depth in embedding
            
            # Execution properties: [revert, log(input_size), log(output_size), log(gas_used)]
            exec_properties[i, 0] = event['has_revert']
            exec_properties[i, 1] = math.log(event['input_size'] + 1) / math.log(10000)
            exec_properties[i, 2] = math.log(event['output_size'] + 1) / math.log(10000)
            exec_properties[i, 3] = math.log(event['gas_used'] + 1) / math.log(10e6)
        
        return call_type_ids, contract_ids, func_selector_ids, depths, exec_properties
    
    def _empty_sequence(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return empty sequence arrays."""
        max_len = self.max_sequence_length
        return (
            np.zeros(max_len, dtype=np.int32),
            np.zeros(max_len, dtype=np.int32),
            np.zeros(max_len, dtype=np.int32),
            np.zeros(max_len, dtype=np.int32),
            np.zeros((max_len, 4), dtype=np.float32)
        )


class EdgeFeatureMask(nn.Module):
    """Create attention mask for padded sequences."""
    
    @staticmethod
    def create_mask(
        call_type_ids: np.ndarray,
        valid_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Create boolean mask for valid (non-padded) positions.
        
        Args:
            call_type_ids: (seq_len,) integer IDs
            valid_length: If provided, only first valid_length positions are True
        
        Returns:
            (seq_len,) boolean mask (True for valid positions)
        """
        mask = call_type_ids != 0
        
        if valid_length is not None:
            mask[:valid_length] = True
            mask[valid_length:] = False
        
        return mask


class EdgeFeatureExtractor(nn.Module):
    """
    Main edge feature extractor combining external + internal features.
    
    Produces feature vectors for each transaction edge:
    [external_features (7 dims) ; trace_embedding (128 dims)] → 135 dims total
    """
    
    def __init__(
        self,
        trace_embedding_dim: int = 128,
        max_trace_length: int = 256,
        use_trace: bool = True,
        use_external: bool = True
    ):
        super().__init__()
        
        self.trace_embedding_dim = trace_embedding_dim
        self.max_trace_length = max_trace_length
        self.use_trace = use_trace
        self.use_external = use_external
        
        # Feature extractors
        self.external_features = ExternalTransactionFeatures()
        self.internal_features = InternalCallTraceFeatures(
            max_sequence_length=max_trace_length
        )
        
        # Trace embedding layer (convert call events to embeddings)
        if use_trace:
            self.call_event_embedding = CallEventEmbedding(
                call_type_vocab_size=10,
                contract_vocab_size=50000,
                func_selector_vocab_size=100000,
                depth_max=50,
                embedding_dim=trace_embedding_dim // 5  # 5 semantic dimensions
            )
            # When trace_embedding_dim is not divisible by 5, project to requested dim.
            if self.call_event_embedding.output_dim != trace_embedding_dim:
                self.trace_dim_proj = nn.Linear(self.call_event_embedding.output_dim, trace_embedding_dim)
            else:
                self.trace_dim_proj = nn.Identity()
    
    def extract_edge_features(
        self,
        tx: Dict[str, Any],
        include_mask: bool = False
    ) -> Dict[str, Any]:
        """
        Extract all features for a transaction edge.
        
        Args:
            tx: Transaction dict with external and internal data
            include_mask: Whether to return attention mask for sequence
        
        Returns:
            Dictionary with:
            - external_features: (7,) numpy array
            - call_event_ids: tuple of arrays for embedding layer
            - trace_mask: (max_seq_len,) boolean mask
            - combined_features: (135,) if inference, None if training
        """
        result = {}
        
        # Extract external features
        if self.use_external:
            ext_feats = self.external_features.extract(tx)
            result['external_features'] = ext_feats
        else:
            result['external_features'] = np.zeros(7, dtype=np.float32)
        
        # Extract internal call trace features
        if self.use_trace:
            trace_data = tx.get('trace_data', {})
            call_type_ids, contract_ids, func_sel_ids, depths, exec_props = \
                self.internal_features.extract(trace_data)
            
            result['call_type_ids'] = call_type_ids
            result['contract_ids'] = contract_ids
            result['func_selector_ids'] = func_sel_ids
            result['depths'] = depths
            result['exec_properties'] = exec_props
            
            # Create mask based on first non-zero call type
            mask = EdgeFeatureMask.create_mask(call_type_ids)
            result['trace_mask'] = mask
        else:
            # Dummy trace arrays
            result['call_type_ids'] = np.zeros(self.max_trace_length, dtype=np.int32)
            result['contract_ids'] = np.zeros(self.max_trace_length, dtype=np.int32)
            result['func_selector_ids'] = np.zeros(self.max_trace_length, dtype=np.int32)
            result['depths'] = np.zeros(self.max_trace_length, dtype=np.int32)
            result['exec_properties'] = np.zeros((self.max_trace_length, 4), dtype=np.float32)
            result['trace_mask'] = np.zeros(self.max_trace_length, dtype=bool)
        
        return result
    
    def forward(
        self,
        call_type_ids: torch.Tensor,
        contract_ids: torch.Tensor,
        func_selector_ids: torch.Tensor,
        depths: torch.Tensor,
        exec_properties: torch.Tensor,
        external_features: torch.Tensor,
        trace_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine external + trace features into single edge representation.
        
        Args:
            call_type_ids, contract_ids, etc.: From self.extract_edge_features()
            external_features: (batch_size, 7)
            trace_mask: (batch_size, seq_len) boolean mask
        
        Returns:
            (batch_size, 135) combined edge features
        """
        batch_size = external_features.size(0)
        
        if self.use_trace:
            # Create call event embeddings
            trace_emb = self.call_event_embedding(
                call_type_ids, contract_ids, func_selector_ids, depths, exec_properties
            )  # (batch_size, seq_len, 5*embedding_dim)
            trace_emb = self.trace_dim_proj(trace_emb)
            
            # Aggregate trace embeddings
            if trace_mask is not None:
                trace_mask_exp = trace_mask.unsqueeze(-1).expand_as(trace_emb)
                trace_aggregate = (trace_emb * trace_mask_exp).sum(dim=1) / (trace_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-9)
            else:
                trace_aggregate = trace_emb.mean(dim=1)
            # (batch_size, trace_embedding_dim)
            
            # Combine external + trace features
            combined = torch.cat([external_features, trace_aggregate], dim=1)
        else:
            # Only external features
            trace_placeholder = torch.zeros(
                batch_size, self.trace_embedding_dim,
                device=external_features.device,
                dtype=external_features.dtype
            )
            combined = torch.cat([external_features, trace_placeholder], dim=1)
        
        return combined  # (batch_size, 135)
