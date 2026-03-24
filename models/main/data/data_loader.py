"""
Data Loader: Stream transaction data from PostgreSQL and prepare for training.

Handles:
- PostgreSQL connection and querying
- JSON parsing (logs, trace_data)
- Caching for efficiency
- Mini-batch preparation
"""

import os
import json
import psycopg2
import pandas as pd
import pickle
from typing import Iterator, Dict, Optional, List, Tuple, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm


class TransactionDataLoader:
    """Load transaction data from PostgreSQL."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "tx_joined_4000000_4010000"
    ):
        """
        Args:
            Connection parameters (if None, read from .env)
            table_name: Database table to query
        """
        from dotenv import load_dotenv
        load_dotenv()
        
        self.host = host or os.getenv("POSTGRESQL_HOST")
        self.port = int(port or os.getenv("POSTGRESQL_PORT", 5432))
        self.database = database or os.getenv("POSTGRESQL_DATABASE")
        self.user = user or os.getenv("POSTGRESQL_USER")
        self.password = password or os.getenv("POSTGRESQL_PASSWORD")
        self.table_name = table_name
    
    def connect(self):
        """Create database connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
    
    def stream_transactions(
        self,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        chunk_size: int = 5000
    ) -> Iterator[pd.DataFrame]:
        """
        Stream transactions in chunks from database.
        
        Args:
            start_block, end_block: Block range filter
            chunk_size: Number of rows per chunk
        
        Yields:
            Pandas DataFrames with parsed JSON columns
        """
        conn = self.connect()
        
        query = f"""
            SELECT
                block_number,
                transaction_hash,
                transaction_index,
                from_address,
                to_address,
                gas_used,
                value,
                logs,
                trace_data,
                timestamp,
                is_suspicious
            FROM {self.table_name} AS tj
            -- Filter transactions that related to erc20
            WHERE EXISTS (
  SELECT 1
  FROM jsonb_array_elements(COALESCE(tj.logs, '[]'::jsonb)) AS log
  WHERE 
    log ? 'topics'
    AND jsonb_array_length(log->'topics') > 0
    AND log->'topics'->>0 IN (
      '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef', -- Transfer
      '0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925'  -- Approval
    )
)

        """
        
        if start_block is not None and end_block is not None:
            query += f" AND tj.block_number BETWEEN {start_block} AND {end_block}"
        
        query += " ORDER BY tj.timestamp ASC"
        
        try:
            for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
                # Parse JSON columns
                chunk['logs'] = chunk['logs'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else (x or [])
                )
                chunk['trace_data'] = chunk['trace_data'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else (x or {})
                )
                
                yield chunk
        finally:
            print("Closing database connection...")
            conn.close()
    
    def get_transaction_count(
        self,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None
    ) -> int:
        """Get total transaction count in range."""
        conn = self.connect()
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        
        if start_block is not None and end_block is not None:
            query += f" WHERE block_number BETWEEN {start_block} AND {end_block}"
        
        try:
            result = pd.read_sql(query, conn)
            return result.iloc[0, 0]
        finally:
            conn.close()
    
    def get_label_distribution(
        self,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None
    ) -> Dict[str, int]:
        """Check label distribution."""
        conn = self.connect()
        query = f"SELECT is_suspicious, COUNT(*) FROM {self.table_name}"
        
        if start_block is not None and end_block is not None:
            query += f" WHERE block_number BETWEEN {start_block} AND {end_block}"
        
        query += " GROUP BY is_suspicious"
        
        try:
            result = pd.read_sql(query, conn)
            return dict(zip(result['is_suspicious'], result['count']))
        finally:
            conn.close()


class CachedDataLoader:
    """Caching wrapper for efficient data loading."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, name: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{name}.pkl"
    
    def save_cache(self, name: str, data: Any) -> None:
        """Save data to cache."""
        cache_path = self.get_cache_path(name)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_cache(self, name: str) -> Optional[Any]:
        """Load data from cache."""
        cache_path = self.get_cache_path(name)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None


def stream_prepared_transactions(
    data_loader: TransactionDataLoader,
    feature_extractor,
    start_block: int = 4000000,
    end_block: int = 4010000,
    chunk_size: int = 1000,
    show_progress: bool = True
) -> Iterator[Dict[str, Any]]:
    """
    Stream transactions with extracted features.
    
    Args:
        data_loader: TransactionDataLoader instance
        feature_extractor: EdgeFeatureExtractor instance
        start_block, end_block: Block range
        chunk_size: Transactions per batch
        show_progress: Show tqdm progress bar
    
    Yields:
        Dictionary with transaction data + extracted features
    """
    iterator = data_loader.stream_transactions(start_block, end_block, chunk_size)
    
    if show_progress:
        total = data_loader.get_transaction_count(start_block, end_block)
        iterator = tqdm(iterator, total=total // chunk_size, desc="Loading transactions")
    
    for chunk in iterator:
        for _, row in chunk.iterrows():
            # Extract features
            edge_features_dict = feature_extractor.extract_edge_features(row.to_dict())
            
            # Prepare transaction record
            tx_record = {
                'block_number': row['block_number'],
                'transaction_hash': row['transaction_hash'],
                'from_address': row['from_address'],
                'to_address': row['to_address'],
                'timestamp': row['timestamp'],
                'is_suspicious': row.get('is_suspicious', 0),
                'external_features': edge_features_dict['external_features'],
                'call_type_ids': edge_features_dict['call_type_ids'],
                'contract_ids': edge_features_dict['contract_ids'],
                'func_selector_ids': edge_features_dict['func_selector_ids'],
                'depths': edge_features_dict['depths'],
                'status_ids': edge_features_dict['status_ids'],
                'input_sizes': edge_features_dict['input_sizes'],
                'output_sizes': edge_features_dict['output_sizes'],
                'gas_vals': edge_features_dict['gas_vals'],
                'trace_mask': edge_features_dict['trace_mask']
            }
            
            yield tx_record
