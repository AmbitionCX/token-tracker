import { provider } from './provider.js';
import {
    get_tx_receipt,
    find_deployment_block
} from './callBlockInfo.js';
import { getTokenName } from './getTokenName.js';
import { query, insert, getMaxBlockNumber } from './postgresql.js';

const CONTRACT_ADDRESS = process.argv[2] || null;

/* =========================
   建表
========================= */
async function setupTokenTable(contractAddress) {
    const { name, symbol } = await getTokenName(contractAddress);
    console.log(`Token: ${name} (${symbol})`);

    const tableName = `${symbol.toLowerCase()}_transactions`;

    const sql = `
        CREATE TABLE IF NOT EXISTS ${tableName} (
            id SERIAL PRIMARY KEY,
            block_number BIGINT NOT NULL,
            transaction_hash VARCHAR(66) UNIQUE NOT NULL,
            transaction_index INTEGER,
            from_address VARCHAR(42) NOT NULL,
            to_address VARCHAR(42),
            contract_address VARCHAR(42),
            gas_used BIGINT,
            cumulative_gas_used BIGINT,
            gas_price BIGINT,
            logs JSONB,
            timestamp TIMESTAMP
        )
    `;

    await query(sql);
    console.log(`Table '${tableName}' verified`);
    return tableName;
}

/* =========================
   存交易（只存真正相关的）
========================= */
async function storeTxReceipt(txHash, contractAddress, tableName) {
    try {
        const receipt = await get_tx_receipt(txHash);
        if (!receipt || !receipt.logs) return false;

        const target = contractAddress.toLowerCase();

        // 二次校验（理论上 logs 扫描已经保证相关）
        const related = receipt.logs.some(
            log => log.address && log.address.toLowerCase() === target
        );
        if (!related) return false;

        const block = await provider.getBlock(receipt.blockNumber);

        const txData = {
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

        await insert(tableName, txData);
        return true;
    } catch (e) {
        // duplicate tx_hash
        if (e.code === '23505') return false;
        throw e;
    }
}

/* =========================
   基于 eth_getLogs 的高速扫描
========================= */
async function scanByLogs(contractAddress, startBlock, endBlock = 'latest', tableName) {
    const currentBlock = await provider.getBlockNumber();
    const targetEnd = endBlock === 'latest' ? currentBlock : endBlock;

    const BATCH_SIZE = 5000;
    let count = 0;

    for (let from = startBlock; from <= targetEnd; from += BATCH_SIZE) {
        const to = Math.min(from + BATCH_SIZE - 1, targetEnd);

        console.log(`Scanning logs: ${from} → ${to}`);

        let logs;
        try {
            logs = await provider.getLogs({
                address: contractAddress,
                fromBlock: from,
                toBlock: to
            });
        } catch (e) {
            console.error(`getLogs failed [${from}-${to}]`, e.message);
            continue;
        }

        if (logs.length === 0) continue;

        // 一个 tx 可能有多个 log，去重
        const txHashes = [...new Set(logs.map(l => l.transactionHash))];

        for (const txHash of txHashes) {
            const inserted = await storeTxReceipt(
                txHash,
                contractAddress,
                tableName
            );
            if (inserted) count++;
        }
    }

    return count;
}

/* =========================
   主流程
========================= */
async function getTokenTransactions(contractAddress = CONTRACT_ADDRESS) {
    if (!contractAddress) {
        console.error('Contract address required');
        process.exit(1);
    }

    const tableName = await setupTokenTable(contractAddress);
    const { isEmpty, maxBlock } = await getMaxBlockNumber(tableName);

    let startBlock;
    if (isEmpty) {
        console.log('Table empty, finding deployment block...');
        startBlock = await find_deployment_block(contractAddress);
    } else {
        startBlock = Number(maxBlock) + 1;
        console.log(`Resume from block ${startBlock}`);
    }

    const currentBlock = await provider.getBlockNumber();
    if (startBlock > currentBlock) {
        console.log('No new blocks');
        return 0;
    }

    const count = await scanByLogs(
        contractAddress,
        startBlock,
        'latest',
        tableName
    );

    console.log(`Scan finished. Inserted ${count} new transactions`);
    return count;
}

/* ========================= */

getTokenTransactions(CONTRACT_ADDRESS)
    .then(() => process.exit(0))
    .catch(err => {
        console.error(err);
        process.exit(1);
    });
