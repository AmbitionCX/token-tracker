import pLimit from "p-limit";
import { provider } from "./provider.js";
import { getTokenName } from "./getTokenName.js";
import { query } from "./postgresql.js";

const CONTRACT_ADDRESS = process.argv[2] || null;

/* ---------- 参数 ---------- */

const TX_BATCH = 400;
const TRACE_CONCURRENCY = 12;
const INSERT_BATCH = 200;

/* ---------- profiling ---------- */

const stats = {
    txFetched: 0,
    tracesDone: 0,
    dbFetchTime: 0,
    traceTime: 0,
    dbInsertTime: 0
};

function printStats() {

    console.log("\n------------ PROFILING ------------");

    console.log("tx fetched:", stats.txFetched);
    console.log("traces done:", stats.tracesDone);

    console.log(
        "avg DB fetch:",
        stats.txFetched ? (stats.dbFetchTime / stats.txFetched).toFixed(2) : 0,
        "ms"
    );

    console.log(
        "avg trace RPC:",
        stats.tracesDone ? (stats.traceTime / stats.tracesDone).toFixed(2) : 0,
        "ms"
    );

    console.log(
        "avg DB insert:",
        stats.tracesDone ? (stats.dbInsertTime / stats.tracesDone).toFixed(2) : 0,
        "ms"
    );

    const throughput = (stats.tracesDone / 10).toFixed(2);

    console.log("throughput:", throughput, "tx/s");

    console.log("-----------------------------------\n");

    /* reset every 10s */

    stats.txFetched = 0;
    stats.tracesDone = 0;
    stats.dbFetchTime = 0;
    stats.traceTime = 0;
    stats.dbInsertTime = 0;
}

setInterval(printStats, 10000);

/* ---------- trace RPC ---------- */

async function getTxTrace(txHash) {

    const t0 = Date.now();

    try {

        const trace = await provider.send("debug_traceTransaction", [
            txHash,
            {
                tracer: "callTracer",
                reexec: 14000000
            }
        ]);

        const dt = Date.now() - t0;

        stats.traceTime += dt;
        stats.tracesDone++;

        return trace;

    } catch (err) {

        console.error(`Trace failed: ${txHash}`, err.message);
        return null;
    }
}

/* ---------- DB setup ---------- */

async function setupTraceTable(symbol) {

    const table = `${symbol.toLowerCase()}_traces`;

    const sql = `
        CREATE TABLE IF NOT EXISTS ${table} (
            id SERIAL PRIMARY KEY,
            transaction_hash VARCHAR(66) UNIQUE,
            trace JSONB
        )
    `;

    await query(sql);

    return table;
}

/* ---------- last processed id ---------- */

async function getLastProcessedId(traceTable) {

    const res = await query(`
        SELECT MAX(id) as last_id
        FROM ${traceTable}
    `);

    return res.rows[0].last_id || 0;
}

/* ---------- get tx batch ---------- */

async function getNextTransactions(symbol, lastId) {

    const t0 = Date.now();

    const table = `${symbol.toLowerCase()}_transactions`;

    const sql = `
        SELECT id, transaction_hash
        FROM ${table}
        WHERE id > $1
        ORDER BY id ASC
        LIMIT $2
    `;

    const res = await query(sql, [lastId, TX_BATCH]);

    const dt = Date.now() - t0;

    stats.dbFetchTime += dt;
    stats.txFetched += res.rows.length;

    return res.rows;
}

/* ---------- batch insert ---------- */

async function batchInsert(table, rows) {

    if (rows.length === 0) return;

    const t0 = Date.now();

    const values = [];
    const placeholders = [];

    rows.forEach((r, i) => {

        const idx = i * 2;

        placeholders.push(`($${idx + 1}, $${idx + 2})`);

        values.push(
            r.transaction_hash,
            JSON.stringify(r.trace)
        );
    });

    const sql = `
        INSERT INTO ${table} (transaction_hash, trace)
        VALUES ${placeholders.join(",")}
        ON CONFLICT (transaction_hash) DO NOTHING
    `;

    await query(sql, values);

    const dt = Date.now() - t0;

    stats.dbInsertTime += dt;
}

/* ---------- main pipeline ---------- */

async function fetchTraces(contractAddress) {

    const { symbol } = await getTokenName(contractAddress);

    const traceTable = await setupTraceTable(symbol);

    let currentId = await getLastProcessedId(traceTable);

    console.log(`Start tracing from id ${currentId}`);

    const limit = pLimit(TRACE_CONCURRENCY);

    let total = 0;

    while (true) {

        const txs = await getNextTransactions(symbol, currentId);

        if (txs.length === 0) {
            console.log("No more transactions.");
            break;
        }

        console.log(`Processing ${txs.length} transactions`);

        /* ---- parallel trace ---- */

        const tasks = txs.map(tx =>
            limit(async () => {

                const trace = await getTxTrace(tx.transaction_hash);

                if (!trace) return null;

                return {
                    transaction_hash: tx.transaction_hash,
                    trace
                };
            })
        );

        const results = await Promise.all(tasks);

        const valid = results.filter(Boolean);

        /* ---- batch insert ---- */

        for (let i = 0; i < valid.length; i += INSERT_BATCH) {

            const chunk = valid.slice(i, i + INSERT_BATCH);

            await batchInsert(traceTable, chunk);
        }

        currentId = txs[txs.length - 1].id;

        total += valid.length;

        console.log(`Stored ${valid.length} traces | total ${total}`);
    }

    console.log(`Trace scan finished: ${total}`);
}

/* ---------- main ---------- */

async function main() {

    if (!CONTRACT_ADDRESS) {
        console.error("Contract address required");
        process.exit(1);
    }

    await fetchTraces(CONTRACT_ADDRESS);
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});