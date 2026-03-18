import os
import json
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# 读取.env
load_dotenv()

def stream_transactions(start_block=None, end_block=None, chunk_size=5000):
    """
    Streaming transactions from PostgreSQL
    """

    conn = psycopg2.connect(
        host=os.getenv("POSTGRESQL_HOST"),
        port=os.getenv("POSTGRESQL_PORT"),
        database=os.getenv("POSTGRESQL_DATABASE"),
        user=os.getenv("POSTGRESQL_USER"),
        password=os.getenv("POSTGRESQL_PASSWORD"),
    )

    query = """
    SELECT
        block_number,
        transaction_hash,
        transaction_index,
        from_address,
        to_address,
        gas_used,
        gas_price,
        logs,
        trace_data,
        timestamp
    FROM tx_joined_10000000_10001000
    """
#if need to modify the table name, please change it here. 

    if start_block and end_block:
        query += f" WHERE block_number BETWEEN {start_block} AND {end_block}"

    for chunk in pd.read_sql(query, conn, chunksize=chunk_size):

        # 解析 JSON
        chunk["logs"] = chunk["logs"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        chunk["trace_data"] = chunk["trace_data"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        yield chunk

    conn.close()