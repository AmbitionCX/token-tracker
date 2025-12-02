import json
from eth_utils import function_signature_to_4byte_selector
from eth_abi import decode_abi

def build_abi_maps(abi_json):
    """
    构建：
      selector_map: selector → function signature
      abi_inputs_map: selector → [ {name,type}, ... ]
    """
    selector_map = {}
    inputs_map = {}

    for item in abi_json:
        if item.get("type") != "function":
            continue

        name = item["name"]
        inputs = item["inputs"]

        arg_types = ",".join(i["type"] for i in inputs)
        signature = f"{name}({arg_types})"

        selector = function_signature_to_4byte_selector(signature).hex()
        selector = "0x" + selector

        selector_map[selector] = signature
        inputs_map[selector] = inputs

    return selector_map, inputs_map


def decode_function_call(input_hex, abi_inputs):
    """
    input_hex = "0x095ea7..."
    abi_inputs = ABI 中的 inputs 列表
    """
    if not input_hex or len(input_hex) < 10:
        return None, None

    selector = input_hex[:10]
    data_hex = input_hex[10:]

    if len(data_hex) % 2 == 1:  # 奇数字节不合法
        return None, None

    types = [inp["type"] for inp in abi_inputs]

    try:
        decoded = decode_abi(types, bytes.fromhex(data_hex))
        return selector, decoded
    except Exception:
        return selector, None


def format_function(signature, abi_inputs, decoded_params):
    """
    输出格式：
      approve(address: 0x123..., uint256: 100000)
    """
    if decoded_params is None:
        return signature + " (DECODE_FAILED)"

    args = []
    for (inp, val) in zip(abi_inputs, decoded_params):
        args.append(f"{inp['name'] or inp['type']}: {val}")

    inside = ", ".join(args)
    func_name = signature.split("(")[0]
    return f"{func_name}({inside})"
