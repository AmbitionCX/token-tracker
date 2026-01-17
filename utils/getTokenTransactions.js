import pLimit from 'p-limit';
import { provider } from './provider.js';
import { get_tx_receipt, find_deployment_block } from './callBlockInfo.js';
import { getTokenName } from './getTokenName.js';
import { query, getMaxBlockNumber } from './postgresql.js';

const CONTRACT_ADDRESS = process.argv[2];
const BATCH_SIZE = 5000;     // block batch
const INSERT_BATCH = 300;   // db insert batch
const RPC_CONCURRENCY = 10; // receipt concurrency

/* =========================
   工具：批量 INSERT
========================= */
async function batchInsert(tableName, rows) {
    if (rows.length === 0) return;

    const cols = Object.keys(rows[0]);
    const values = [];
    const placeholders = [];

    rows.forEach((row, i) => {
        const offset = i * cols.length;
        placeholders.push(
            `(${cols.map((_, j) => `$${offset + j + 1}`).join(',')})`
        );
        values.push(...cols.map(c => row[c]));
    });

    const sql = `
        INSERT INTO ${tableName} (${cols.join(',')})
        VALUES ${placeholders.join(',')}
        ON CONFLICT (transaction_hash) DO NOTHING
    `;

    await query(sql, values);
}

/* =========================
   主逻辑
========================= */
async function run(contractAddress) {
    if (!contractAddress) {
        console.error('Contract address required');
        process.exit(1);
    }

    /* -------- token info -------- */
    const { name, symbol } = await getTokenName(contractAddress);
    console.log(`Token: ${name} (${symbol})`);

    const tableName = `${symbol.toLowerCase()}_transactions`;

    await query(`
        CREATE TABLE IF NOT EXISTS ${tableName} (
            id SERIAL PRIMARY KEY,
            block_number BIGINT,
            transaction_hash VARCHAR(66) UNIQUE,
            transaction_index INTEGER,
            from_address VARCHAR(42),
            to_address VARCHAR(42),
            contract_address VARCHAR(42),
            gas_used BIGINT,
            cumulative_gas_used BIGINT,
            gas_price BIGINT,
            logs JSONB,
            timestamp TIMESTAMP
        )
    `);

    /* -------- resume logic -------- */
    const { isEmpty, maxBlock } = await getMaxBlockNumber(tableName);
    let startBlock = isEmpty
        ? await find_deployment_block(contractAddress)
        : Number(maxBlock) + 1;

    const latest = await provider.getBlockNumber();
    console.log(`Scanning ${startBlock} → ${latest}`);

    /* -------- caches -------- */
    const blockCache = new Map();
    const limit = pLimit(RPC_CONCURRENCY);

    async function getBlockCached(blockNumber) {
        if (!blockCache.has(blockNumber)) {
            blockCache.set(
                blockNumber,
                await provider.getBlock(blockNumber)
            );
        }
        return blockCache.get(blockNumber);
    }

    async function buildTxData(txHash) {
        const receipt = await get_tx_receipt(txHash);
        if (!receipt || !receipt.logs) return null;

        const target = contractAddress.toLowerCase();
        const related = receipt.logs.some(
            l => l.address && l.address.toLowerCase() === target
        );
        if (!related) return null;

        const block = await getBlockCached(receipt.blockNumber);

        return {
            block_number: Number(receipt.blockNumber),
            transaction_hash: receipt.hash,
            transaction_index: Number(receipt.index),
            from_address: receipt.from,
            to_address: receipt.to,
            contract_address: contractAddress,
            gas_used: Number(receipt.gasUsed),
            cumulative_gas_used: Number(receipt.cumulativeGasUsed),
            gas_price: receipt.effectiveGasPrice
                ? Number(receipt.effectiveGasPrice)
                : null,
            logs: JSON.stringify(receipt.logs),
            timestamp: new Date(block.timestamp * 1000)
        };
    }

    /* -------- scan -------- */
    for (let from = startBlock; from <= latest; from += BATCH_SIZE) {
        const to = Math.min(from + BATCH_SIZE - 1, latest);
        console.log(`Blocks ${from} → ${to}`);

        let logs;
        try {
            logs = await provider.getLogs({
                address: contractAddress,
                fromBlock: from,
                toBlock: to
            });
        } catch (e) {
            console.error('getLogs failed:', e.message);
            continue;
        }

        if (logs.length === 0) continue;

        const txHashes = [...new Set(logs.map(l => l.transactionHash))];

        const tasks = txHashes.map(h =>
            limit(() => buildTxData(h))
        );

        const results = await Promise.all(tasks);
        const rows = results.filter(Boolean);

        for (let i = 0; i < rows.length; i += INSERT_BATCH) {
            const chunk = rows.slice(i, i + INSERT_BATCH);
            await batchInsert(tableName, chunk);
        }

        console.log(`Inserted ${rows.length} txs`);
    }

    console.log('Scan finished');
}

/* ========================= */

run(CONTRACT_ADDRESS)
    .then(() => process.exit(0))
    .catch(err => {
        console.error(err);
        process.exit(1);
    });
