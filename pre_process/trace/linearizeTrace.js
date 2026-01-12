/**
 * DFS 线性化 call trace
 * @param {Object} trace - transaction trace root
 * @returns {Array<Object>} linearized call events
 */
export function linearizeCallTrace(traceRoot) {
  const sequence = [];

  function dfs(node, depth = 0, parentId = -1) {
    if (!node) return;

    const id = sequence.length;

    const input = node.input ?? "0x";
    const output = node.output ?? "0x";

    sequence.push({
      id,
      parentId,
      depth,

      callType: node.type ?? "CALL",
      callee: node.to ?? null,
      selector: input.length >= 10 ? input.slice(0, 10) : null,

      reverted: Boolean(node.error || node.revertReason),
      inputLength: Math.max(0, (input.length - 2) / 2),
      outputLength: Math.max(0, (output.length - 2) / 2),
      gasUsed: node.gasUsed ? parseInt(node.gasUsed, 16) : 0,
    });

    if (Array.isArray(node.calls)) {
      for (const child of node.calls) {
        dfs(child, depth + 1, id);
      }
    }
  }

  dfs(traceRoot);
  return sequence;
}
