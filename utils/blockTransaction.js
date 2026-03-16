// 用法: node blockTransaction.js 10000000 10001000

import fs from "fs";
import pLimit from "p-limit";
import { provider } from "./provider.js";
import { query } from "./postgresql.js";

const START_BLOCK = parseInt(process.argv[2]);
const END_BLOCK = parseInt(process.argv[3]);

if (!START_BLOCK || !END_BLOCK || START_BLOCK > END_BLOCK) {
    console.error("Usage: node blockTransaction.js <startBlock> <endBlock>");
    process.exit(1);
}

const TABLE_NAME = `block_transactions_${START_BLOCK}_${END_BLOCK}`;
const PROGRESS_FILE = "progress_transactions.json";

/* ---------------- progress ---------------- */

let lastProcessedBlock = START_BLOCK - 1;

if (fs.existsSync(PROGRESS_FILE)) {
    const progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, "utf8"));
    lastProcessedBlock = progress.lastProcessedBlock || lastProcessedBlock;
}

function saveProgress(block) {
    fs.writeFileSync(PROGRESS_FILE, JSON.stringify({ lastProcessedBlock: block }));
}

/* ---------------- concurrency ---------------- */

const BLOCK_CONCURRENCY = 3;
const RECEIPT_CONCURRENCY = 20;

const blockLimit = pLimit(BLOCK_CONCURRENCY);
const receiptLimit = pLimit(RECEIPT_CONCURRENCY);

/* ---------------- DB queue ---------------- */

const DB_BATCH_SIZE = 500;
const dbQueue = [];
let dbInserting = false;

/* ---------------- stats ---------------- */

const stats = {
    blocks: 0,
    txs: 0,
    receipts: 0,
    inserts: 0
};

function printStats() {
    console.log(`
----------- STATS -----------
blocks processed : ${stats.blocks}
tx fetched       : ${stats.txs}
receipts fetched : ${stats.receipts}
db inserts       : ${stats.inserts}
queue            : ${dbQueue.length}
------------------------------
`);
}

setInterval(printStats, 10000);

/* ---------------- helpers ---------------- */

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function safeNumber(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

/* ---------------- RPC helpers ---------------- */

async function getBlock(blockNumber) {

    while (true) {

        try {

            return await provider.getBlock(blockNumber);

        } catch (err) {

            console.error("getBlock retry", blockNumber);
            await sleep(1000);

        }

    }

}

async function getReceipt(txHash) {

    let retries = 0;

    while (true) {

        try {

            const receipt = await provider.getTransactionReceipt(txHash);

            if (receipt) return receipt;

        } catch (err) {
            console.error("receipt error", txHash);
        }

        retries++;

        await sleep(Math.min(500 * retries, 5000));

    }

}

/* ---------------- DB insert ---------------- */

async function insertToDB() {

    if (dbInserting || dbQueue.length === 0) return;

    dbInserting = true;

    const batch = dbQueue.splice(0, DB_BATCH_SIZE);

    try {

        const cols = [
            "block_number",
            "transaction_hash",
            "transaction_index",
            "from_address",
            "to_address",
            "gas_used",
            "gas_price",
            "logs",
            "timestamp"
        ];

        const values = [];
        const placeholders = [];

        batch.forEach((row, i) => {

            const offset = i * cols.length;

            placeholders.push(
                `(${cols.map((_, j) => `$${offset + j + 1}`).join(",")})`
            );

            values.push(
                row.block_number,
                row.transaction_hash,
                row.transaction_index,
                row.from_address,
                row.to_address,
                row.gas_used,
                row.gas_price,
                row.logs,
                row.timestamp
            );

        });

        await query(`
            INSERT INTO ${TABLE_NAME} (${cols.join(",")})
            VALUES ${placeholders.join(",")}
            ON CONFLICT (transaction_hash)
            DO UPDATE SET
                gas_used = EXCLUDED.gas_used
        `, values);

        stats.inserts += batch.length;

    } catch (err) {

        console.error("DB insert error", err);

        dbQueue.unshift(...batch);

        await sleep(2000);

    } finally {

        dbInserting = false;

    }

}

/* ---------------- receipt worker ---------------- */

async function processReceipt(txHash, timestamp) {

    const receipt = await getReceipt(txHash);

    stats.receipts++;

    const txData = {

        block_number: safeNumber(receipt.blockNumber),

        transaction_hash: receipt.hash,

        transaction_index: safeNumber(receipt.index),

        from_address: receipt.from,

        to_address: receipt.to,

        gas_used: safeNumber(receipt.gasUsed),

        gas_price: receipt.effectiveGasPrice
            ? Number(receipt.effectiveGasPrice)
            : null,

        logs: JSON.stringify(receipt.logs),

        timestamp: new Date(Number(timestamp) * 1000)

    };

    dbQueue.push(txData);

    if (dbQueue.length >= DB_BATCH_SIZE) {

        await insertToDB();

    }

}

/* ---------------- block worker ---------------- */

async function processBlock(blockNumber) {

    const block = await getBlock(blockNumber);

    if (!block) return;

    const timestamp = block.timestamp;

    const txHashes = block.transactions;

    stats.txs += txHashes.length;

    const tasks = txHashes.map(hash =>
        receiptLimit(() => processReceipt(hash, timestamp))
    );

    await Promise.all(tasks);

    stats.blocks++;

    saveProgress(blockNumber);

    console.log(`block ${blockNumber} done (${txHashes.length} tx)`);

}

/* ---------------- main ---------------- */

async function main() {

    await query(`
        CREATE TABLE IF NOT EXISTS ${TABLE_NAME} (

            id SERIAL PRIMARY KEY,

            block_number BIGINT,

            transaction_hash VARCHAR(66) UNIQUE,

            transaction_index INT,

            from_address VARCHAR(42),

            to_address VARCHAR(42),

            gas_used BIGINT,

            gas_price BIGINT,

            logs JSONB,

            timestamp TIMESTAMP
        )
    `);

    console.log(`Scanning ${lastProcessedBlock + 1} → ${END_BLOCK}`);

    const tasks = [];

    for (let block = lastProcessedBlock + 1; block <= END_BLOCK; block++) {

        tasks.push(
            blockLimit(() => processBlock(block))
        );

    }

    await Promise.all(tasks);

    while (dbQueue.length > 0) {

        await insertToDB();

    }

    console.log("Pipeline finished");

}

main().catch(console.error);