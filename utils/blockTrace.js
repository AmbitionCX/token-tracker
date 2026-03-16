//用法示例：node utils/blockTracePipeline.js 10000000 10001000
import fs from 'fs';
import pLimit from 'p-limit';
import { provider } from './provider.js';
import { query } from './postgresql.js';

const START_BLOCK = parseInt(process.argv[2]);
const END_BLOCK = parseInt(process.argv[3]);
const PROGRESS_FILE = 'progress_traces.json';
const TABLE_NAME = `block_traces_${START_BLOCK}_${END_BLOCK}`;
console.log('Args:', process.argv[2], process.argv[3]);
console.log('Parsed:', START_BLOCK, END_BLOCK);

if (!START_BLOCK || !END_BLOCK || START_BLOCK > END_BLOCK) {
    console.error('Usage: node blockTracePipeline.js <startBlock> <endBlock>');
    process.exit(1);
}

// Load progress
let lastProcessedBlock = START_BLOCK - 1;
if (fs.existsSync(PROGRESS_FILE)) {
    const progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
    lastProcessedBlock = progress.lastProcessedBlock || lastProcessedBlock;
}

// Worker queue with concurrency
const WORKER_CONCURRENCY = 1; // Adjust based on geth rate limits
const limit = pLimit(WORKER_CONCURRENCY);

// DB writer queue for backpressure
const DB_BATCH_SIZE = 1000;
const dbQueue = [];
let dbInserting = false;

// Stats
const stats = {
    blocksProcessed: 0,
    tracesFetched: 0,
    dbInserts: 0,
    totalTime: 0
};

function printStats() {
    console.log(`Blocks processed: ${stats.blocksProcessed}, Traces: ${stats.tracesFetched}, DB inserts: ${stats.dbInserts}, Avg time/block: ${(stats.totalTime / Math.max(stats.blocksProcessed, 1)).toFixed(2)} ms`);
}

setInterval(printStats, 10000);

function saveProgress(block) {
    fs.writeFileSync(PROGRESS_FILE, JSON.stringify({ lastProcessedBlock: block }));
}

async function getBlockTrace(blockNumber, retries = 3) {
    const t0 = Date.now();
    for (let i = 0; i < retries; i++) {
        try {
            const trace = await provider.send('debug_traceBlockByNumber', [
                `0x${blockNumber.toString(16)}`,
                { tracer: 'callTracer' }
            ]);
            const dt = Date.now() - t0;
            stats.totalTime += dt;
            return trace;
        } catch (err) {
            console.error(`Trace failed for block ${blockNumber} (attempt ${i+1}):`, err.message);
            if (i < retries - 1) await new Promise(res => setTimeout(res, 1000 * (i + 1))); // Exponential backoff
        }
    }
    return null;
}

async function processBlock(blockNumber) {
    const trace = await getBlockTrace(blockNumber);
    if (!trace) {
        console.log(`No trace for block ${blockNumber}`);
        return;
    }

    // Parse traces: trace is array of transaction traces
    const parsedTraces = [];
    for (const txTrace of trace) {
        if (txTrace.result) {
            // Flatten the call tree or extract relevant data
            // For simplicity, store the entire trace
            parsedTraces.push({
                blockNumber,
                txHash: txTrace.txHash,
                trace: JSON.stringify(txTrace.result)
            });
        }
    }

    stats.tracesFetched += parsedTraces.length;

    // Add to DB queue
    dbQueue.push(...parsedTraces);

    // Trigger DB insert if batch size reached
    if (dbQueue.length >= DB_BATCH_SIZE) {
        await insertToDB();
    }

    stats.blocksProcessed++;
    saveProgress(blockNumber);
    console.log(`Processed block ${blockNumber}, traces: ${parsedTraces.length}`);
}

async function insertToDB() {
    if (dbInserting || dbQueue.length === 0) return;
    dbInserting = true;

    const batch = dbQueue.splice(0, DB_BATCH_SIZE);
    try {
        // Use bulk insert
        const values = batch.map((t, i) => `($${i*3+1}, $${i*3+2}, $${i*3+3})`).join(',');
        const params = [];
        batch.forEach(t => {
            params.push(t.blockNumber, t.txHash, t.trace);
        });

        await query(`
            INSERT INTO ${TABLE_NAME} (block_number, tx_hash, trace_data)
            VALUES ${values}
            ON CONFLICT (block_number, tx_hash) DO NOTHING
        `, params);

        stats.dbInserts += batch.length;
    } catch (err) {
        console.error('DB insert failed:', err);
        // Re-add to queue
        dbQueue.unshift(...batch);
    } finally {
        dbInserting = false;
    }
}

async function main() {
    console.log(`lastProcessedBlock: ${lastProcessedBlock}`);
    // Setup DB table if needed
    await query(`
        CREATE TABLE IF NOT EXISTS ${TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            block_number BIGINT NOT NULL,
            tx_hash VARCHAR(66) NOT NULL,
            trace_data JSONB,
            UNIQUE(block_number, tx_hash)
        );
    `);

    console.log(`Starting pipeline from ${lastProcessedBlock + 1} to ${END_BLOCK}`);
    const promises = [];
    for (let block = lastProcessedBlock + 1; block <= END_BLOCK; block++) {
        promises.push(limit(() => processBlock(block)));
    }
    console.log(`Total promises: ${promises.length}`);

    await Promise.all(promises);

    // Final DB insert
    await insertToDB();

    console.log('Pipeline completed.');
    console.log(`Blocks processed: ${stats.blocksProcessed}`);
    console.log(`Traces fetched: ${stats.tracesFetched}`);
    console.log(`DB inserts: ${stats.dbInserts}`);
    console.log(`Avg time per block: ${(stats.totalTime / stats.blocksProcessed).toFixed(2)} ms`);
}

main().catch(console.error);