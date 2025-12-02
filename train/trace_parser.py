from abi_decoder import decode_function_call, format_function

def parse_trace(node, selector_map, abi_inputs_map):
    """
    返回：["approve(spender: 0x..., value: 123)", ...]
    """
    results = []

    if not isinstance(node, dict):
        return results

    inp = node.get("input")
    if isinstance(inp, str) and inp.startswith("0x") and len(inp) >= 10:
        
        selector = inp[:10]
        signature = selector_map.get(selector, f"UNKNOWN_{selector}")
        abi_inputs = abi_inputs_map.get(selector)

        if abi_inputs:
            _, decoded = decode_function_call(inp, abi_inputs)
            formatted = format_function(signature, abi_inputs, decoded)
        else:
            formatted = signature

        results.append(formatted)

    subcalls = node.get("calls")
    if isinstance(subcalls, list):
        for c in subcalls:
            results.extend(parse_trace(c, selector_map, abi_inputs_map))

    return results
