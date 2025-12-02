import os
import json
import pandas as pd
from dotenv import load_dotenv

from etherscan import fetch_token_abi
from abi_decoder import build_abi_maps
from trace_parser import parse_trace
from db import get_engine, load_traces

load_dotenv()

def process_token(token_address: str, output_csv="function_sequence.csv"):

    api_key = os.getenv("ETHERSCAN_API_KEY")

    print(f"Fetching ABI for {token_address} ...")
    abi_json = fetch_token_abi(token_address, api_key)

    selector_map, abi_inputs_map = build_abi_maps(abi_json)
    print(f"Loaded {len(selector_map)} functions.")

    engine = get_engine(
        os.getenv("POSTGRESQL_USER"),
        os.getenv("POSTGRESQL_PASSWORD"),
        os.getenv("POSTGRESQL_HOST"),
        os.getenv("POSTGRESQL_PORT"),
        os.getenv("POSTGRESQL_DATABASE"),
    )

    df = load_traces(engine)

    df["function_sequence"] = df["trace"].apply(
        lambda trace: parse_trace(trace, selector_map, abi_inputs_map)
    )

    df.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")


if __name__ == "__main__":
    token = "0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490"
    process_token(token)
