//用的是本地的trace数据来做测试，后面改成连数据库的版本
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { buildCallSequence } from "./trace/buildCallSequence.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rawPath = path.resolve(
  __dirname,
  "../../Results/recent_100_blocks_traces.json"
);

const outPath = path.resolve(
  __dirname,
  "../../Results/first_tx_calls.json"
);

const data = JSON.parse(fs.readFileSync(rawPath, "utf8"));

const processed = data
  .map(buildCallSequence)
  .filter(Boolean);

const firstTx = processed[0];

fs.writeFileSync(
  outPath,
  JSON.stringify(firstTx, null, 2),
  "utf8"
);

console.log("Saved:", outPath);
