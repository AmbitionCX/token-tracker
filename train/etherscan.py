import requests
import json

def fetch_token_abi(token_address: str, api_key: str):
    """
    调用 Etherscan v2 获取 ABI → 返回解析后的 JSON
    """
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": 1,
        "module": "contract",
        "action": "getabi",
        "address": token_address,
        "apikey": api_key
    }
    r = requests.get(url, params=params)
    data = r.json()

    if data.get("status") != "1":
        raise RuntimeError(f"ABI 获取失败: {data}")

    return json.loads(data["result"])
