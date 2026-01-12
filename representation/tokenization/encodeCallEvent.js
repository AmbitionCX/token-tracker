// representation/tokenization/encodeCallEvent.js

import { encodeExecFeature } from "./discretize.js";

export function encodeCallEvent(c, vocabs) {
  const {
    typeVocab,
    contractVocab,
    funcVocab,
  } = vocabs;

  const exec = encodeExecFeature({
    reverted: c.reverted,
    inputLength: c.inputLength,
    outputLength: c.outputLength,
    gasUsed: c.gasUsed,
  });

  return {
    typeId: typeVocab.getId(c.callType),
    contractId: contractVocab.getId(c.callee),
    funcId: funcVocab.getId(c.selector),
    depthId: c.depth,     // 直接用整数
    exec,                 // 组合特征
  };
}
