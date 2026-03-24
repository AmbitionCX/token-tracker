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
    - value: ETH transfer amount (log-normalised)
    - gas_used: Actual gas consumed (log-normalised)
    - calldata_length: Byte length of transaction input (log-normalised)
    - is_revert: Whether the transaction failed (binary)
    """

    FEATURE_NAMES = [
        'value',
        'gas_used',
        'calldata_length',
        'is_revert',
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
                - trace_data: Root call trace dict

        Returns:
            np.ndarray of shape (4,) with normalised features
        """
        features = []

        # 1. Value: log-normalised
        value = float(tx.get('value', 0) or 0)
        features.append(self._normalize_value(value))

        # 2. Gas used: log-normalised
        gas_used = float(tx.get('gas_used', 0) or 0)
        features.append(self._normalize_value(gas_used, max_val=10e6))

        # 3. Calldata length: byte length of root input, log-normalised
        calldata_length = 0
        trace_data = tx.get('trace_data')
        if isinstance(trace_data, dict) and 'input' in trace_data:
            input_hex = trace_data['input']
            if isinstance(input_hex, str) and input_hex.startswith('0x'):
                calldata_length = (len(input_hex) - 2) // 2
        features.append(self._normalize_value(calldata_length, max_val=10000))

        # 4. Is revert: root output "0x" with no sub-calls → failed tx
        is_revert = 0.0
        if isinstance(trace_data, dict):
            output = trace_data.get('output', '0x')
            if output == '0x' and 'calls' not in trace_data:
                is_revert = 1.0
        features.append(is_revert)

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
            List of linearised call events in execution order.
        """
        events = []

        def dfs(node, current_depth=0):
            inp = node.get('input', '0x') or '0x'
            out = node.get('output', '0x') or '0x'
            gas_raw = node.get('gasUsed', 0)
            event = {
                'type': node.get('type', 'CALL'),
                'to': node.get('to', '0x0'),
                'function_selector': self._extract_function_selector(inp),
                'depth': current_depth,
                'status': 0 if out == '0x' else 1,   # 0 = empty output, 1 = has output
                'input_size': (len(inp) - 2) // 2 if inp.startswith('0x') else 0,
                'output_size': (len(out) - 2) // 2 if out.startswith('0x') else 0,
                'gas_used': int(gas_raw, 16) if isinstance(gas_raw, str) else int(gas_raw or 0),
            }
            events.append(event)

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
    
    def extract(
        self, trace_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and prepare internal call trace features.

        Args:
            trace_data: Raw call trace dict from transaction.

        Returns:
            8-tuple of arrays, each of shape (seq_len,):
            - call_type_ids    (int32)  : call type indices
            - contract_ids     (int32)  : callee address indices
            - func_selector_ids(int32)  : function selector indices
            - depths           (int32)  : call depth
            - status_ids       (int32)  : 0 = empty output, 1 = has output
            - input_sizes      (float32): log(1 + input_byte_length)
            - output_sizes     (float32): log(1 + output_byte_length)
            - gas_vals         (float32): log(1 + gas_used)
        """
        if not isinstance(trace_data, dict):
            return self._empty_sequence()

        events = self._linearize_trace(trace_data)
        if not events:
            return self._empty_sequence()

        events = events[:self.max_sequence_length]
        seq_len = len(events)

        call_type_ids     = np.zeros(seq_len, dtype=np.int32)
        contract_ids      = np.zeros(seq_len, dtype=np.int32)
        func_selector_ids = np.zeros(seq_len, dtype=np.int32)
        depths            = np.zeros(seq_len, dtype=np.int32)
        status_ids        = np.zeros(seq_len, dtype=np.int32)
        input_sizes       = np.zeros(seq_len, dtype=np.float32)
        output_sizes      = np.zeros(seq_len, dtype=np.float32)
        gas_vals          = np.zeros(seq_len, dtype=np.float32)

        func_selector_cache: Dict[str, int] = {}
        next_func_id = 1

        for i, event in enumerate(events):
            call_type_ids[i] = self.CALL_TYPES.get(event['type'], 0)

            contract_addr = event['to'].lower()
            contract_ids[i] = self._get_contract_id(contract_addr) % 50000

            func_sel = event['function_selector']
            if func_sel not in func_selector_cache:
                func_selector_cache[func_sel] = next_func_id
                next_func_id += 1
            func_selector_ids[i] = func_selector_cache[func_sel] % 100000

            depths[i]     = min(event['depth'], 49)
            status_ids[i] = event['status']

            input_sizes[i]  = math.log(1 + event['input_size'])
            output_sizes[i] = math.log(1 + event['output_size'])
            gas_vals[i]     = math.log(1 + event['gas_used'])

        return (call_type_ids, contract_ids, func_selector_ids, depths,
                status_ids, input_sizes, output_sizes, gas_vals)

    def _empty_sequence(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return zero-filled arrays for an empty / missing trace."""
        n = self.max_sequence_length
        return (
            np.zeros(n, dtype=np.int32),    # call_type_ids
            np.zeros(n, dtype=np.int32),    # contract_ids
            np.zeros(n, dtype=np.int32),    # func_selector_ids
            np.zeros(n, dtype=np.int32),    # depths
            np.zeros(n, dtype=np.int32),    # status_ids
            np.zeros(n, dtype=np.float32),  # input_sizes
            np.zeros(n, dtype=np.float32),  # output_sizes
            np.zeros(n, dtype=np.float32),  # gas_vals
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
    [external_features (4 dims) ; trace_embedding (trace_embedding_dim)] total
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

        # Trace embedding layer (convert call events to per-token embeddings)
        if use_trace:
            self.call_event_embedding = CallEventEmbedding(
                call_type_vocab_size=10,
                contract_vocab_size=50000,
                func_selector_vocab_size=100000,
                depth_max=50,
                embedding_dim=trace_embedding_dim // 8  # 8 semantic dimensions
            )
            # Project concatenated token embeddings to trace_embedding_dim if needed.
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
            tx: Transaction dict with external and internal data.
            include_mask: Unused; kept for API compatibility.

        Returns:
            Dictionary with:
            - external_features  : (4,)          numpy array
            - call_type_ids      : (max_seq_len,) int32
            - contract_ids       : (max_seq_len,) int32
            - func_selector_ids  : (max_seq_len,) int32
            - depths             : (max_seq_len,) int32
            - status_ids         : (max_seq_len,) int32
            - input_sizes        : (max_seq_len,) float32
            - output_sizes       : (max_seq_len,) float32
            - gas_vals           : (max_seq_len,) float32
            - trace_mask         : (max_seq_len,) bool
        """
        result = {}

        # External features (4-dim)
        if self.use_external:
            result['external_features'] = self.external_features.extract(tx)
        else:
            result['external_features'] = np.zeros(4, dtype=np.float32)

        # Internal call trace features (8 separate arrays)
        if self.use_trace:
            trace_data = tx.get('trace_data', {})
            (call_type_ids, contract_ids, func_sel_ids, depths,
             status_ids, input_sizes, output_sizes, gas_vals) = \
                self.internal_features.extract(trace_data)

            result['call_type_ids']     = call_type_ids
            result['contract_ids']      = contract_ids
            result['func_selector_ids'] = func_sel_ids
            result['depths']            = depths
            result['status_ids']        = status_ids
            result['input_sizes']       = input_sizes
            result['output_sizes']      = output_sizes
            result['gas_vals']          = gas_vals
            result['trace_mask']        = EdgeFeatureMask.create_mask(call_type_ids)
        else:
            n = self.max_trace_length
            result['call_type_ids']     = np.zeros(n, dtype=np.int32)
            result['contract_ids']      = np.zeros(n, dtype=np.int32)
            result['func_selector_ids'] = np.zeros(n, dtype=np.int32)
            result['depths']            = np.zeros(n, dtype=np.int32)
            result['status_ids']        = np.zeros(n, dtype=np.int32)
            result['input_sizes']       = np.zeros(n, dtype=np.float32)
            result['output_sizes']      = np.zeros(n, dtype=np.float32)
            result['gas_vals']          = np.zeros(n, dtype=np.float32)
            result['trace_mask']        = np.zeros(n, dtype=bool)

        return result
    
    def forward(
        self,
        call_type_ids: torch.Tensor,
        contract_ids: torch.Tensor,
        func_selector_ids: torch.Tensor,
        depths: torch.Tensor,
        status_ids: torch.Tensor,
        input_sizes: torch.Tensor,
        output_sizes: torch.Tensor,
        gas_vals: torch.Tensor,
        external_features: torch.Tensor,
        trace_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine external + trace features into a single edge representation.

        Args:
            call_type_ids      : (batch, seq_len)
            contract_ids       : (batch, seq_len)
            func_selector_ids  : (batch, seq_len)
            depths             : (batch, seq_len)
            status_ids         : (batch, seq_len)
            input_sizes        : (batch, seq_len)  log-normalised
            output_sizes       : (batch, seq_len)  log-normalised
            gas_vals           : (batch, seq_len)  log-normalised
            external_features  : (batch, 4)
            trace_mask         : (batch, seq_len)  bool, True = valid position

        Returns:
            (batch, 4 + trace_embedding_dim) combined edge features
        """
        batch_size = external_features.size(0)

        if self.use_trace:
            # Per-token embeddings: (batch, seq_len, 8*embedding_dim)
            trace_emb = self.call_event_embedding(
                call_type_ids, contract_ids, func_selector_ids, depths,
                status_ids, input_sizes, output_sizes, gas_vals, trace_mask
            )
            trace_emb = self.trace_dim_proj(trace_emb)  # (batch, seq_len, trace_embedding_dim)

            # Mask-aware mean pooling
            if trace_mask is not None:
                mask_exp = trace_mask.unsqueeze(-1).expand_as(trace_emb)
                trace_aggregate = (trace_emb * mask_exp).sum(dim=1) / (
                    trace_mask.sum(dim=1, keepdim=True) + 1e-9
                )
            else:
                trace_aggregate = trace_emb.mean(dim=1)
            # (batch, trace_embedding_dim)

            combined = torch.cat([external_features, trace_aggregate], dim=1)
        else:
            trace_placeholder = torch.zeros(
                batch_size, self.trace_embedding_dim,
                device=external_features.device,
                dtype=external_features.dtype
            )
            combined = torch.cat([external_features, trace_placeholder], dim=1)

        return combined  # (batch, 4 + trace_embedding_dim)
