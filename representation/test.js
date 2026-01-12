import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { buildTokenSequence } from "./tokenization/buildTokens.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 指向 first_tx_calls.json
const txPath = path.resolve(
  __dirname,
  "../../Results/first_tx_calls.json"
);

const tx = JSON.parse(fs.readFileSync(txPath, "utf8"));

const result = buildTokenSequence(tx);

console.log("Tx:", result.txHash);
console.log("L =", result.L);
console.log("First 3 tokens:");
console.dir(result.tokens.slice(0, 3), { depth: null });
//保存到文件
const outPath = path.resolve(__dirname, "../../Results/token_sequence.json");
fs.writeFileSync(outPath, JSON.stringify(result, null, 2));