import { linearizeCallTrace } from "./linearizeTrace.js";

export function buildCallSequence(tx) {
  if (!tx.trace) return null;

  const calls = linearizeCallTrace(tx.trace);

  return {
    blockNumber: tx.blockNumber,
    txHash: tx.txHash,
    L: calls.length,
    C_e: calls,
  };
}

