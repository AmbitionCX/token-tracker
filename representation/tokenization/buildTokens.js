// representation/tokenization/buildTokens.js

import { Vocab } from "./vocab.js";
import { encodeCallEvent } from "./encodeCallEvent.js";

export function buildTokenSequence(tx) {
  if (!Array.isArray(tx.C_e)) {
    throw new Error("buildTokenSequence: tx.C_e is missing or not an array");
  }

  const vocabs = {
    typeVocab: new Vocab(),
    contractVocab: new Vocab(),
    funcVocab: new Vocab(),
  };

  const tokens = tx.C_e.map(c =>
    encodeCallEvent(c, vocabs)
  );

  return {
    txHash: tx.txHash,
    blockNumber: tx.blockNumber,
    L: tokens.length,
    tokens,     // [ x_1, ..., x_L ]
    vocabs,
  };
}
