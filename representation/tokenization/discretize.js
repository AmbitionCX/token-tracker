// representation/tokenization/discretize.js

export function bucketizeGas(gas) {
  if (gas === 0) return 0;
  if (gas < 5_000) return 1;
  if (gas < 50_000) return 2;
  return 3;
}

export function bucketizeLength(len) {
  if (len === 0) return 0;
  if (len <= 32) return 1;
  if (len <= 128) return 2;
  return 3;
}

export function encodeExecFeature({
  reverted,
  inputLength,
  outputLength,
  gasUsed,
}) {
  return {
    reverted: reverted ? 1 : 0,
    gasBucket: bucketizeGas(gasUsed),
    inputBucket: bucketizeLength(inputLength),
    outputBucket: bucketizeLength(outputLength),
  };
}
