import { provider } from './provider.js';
import {
    get_block_info,
    get_tx_receipt,
    find_deployment_block
} from './callBlockInfo.js';
import { getTokenName } from './getTokenName.js';
import { query, insert, getMaxBlockNumber } from './postgresql.js';

const CONTRACT_ADDRESS = process.argv[2]?.toLowerCase() || null;

/* ================= 工具函数 ================= */

// 判断 receipt 是否和 token 有关（核心逻辑）
function receiptHasTokenEvent(receipt, tokenAddress) {
    return receipt.logs.some(
        log => log.address && log.address.toLowerCase() === tokenAddress
    );
}

/* ================= 建表 ================= */

async function setupTokenTable(contractAddress) {
    const { name, symbol } = await getTokenName(contractAddress);
    console.log(`Token: ${name} (${symbol})`);

    const tableName = `${symbol.toLowerCase()}_transactions`;

    await query(`
        CREATE TABLE IF NOT EXISTS ${tableName} (
            id SERIAL PRIMARY KEY,
            block_number BIGINT NOT NULL,
            transaction_hash VARCHAR(66) UNIQUE NOT NULL,
            transaction_index INTEGER,
            from_address VARCHAR(42),
            to_address VARCHAR(42),
            gas_used BIGINT,
            gas_price BIGINT,
            logs JSONB,
            timestamp TIMESTAMP
        )
    `);

    console.log(`Table '${tableName}' verified`);
    return tableName;
}

/* ================= 存交易 ================= */

async function storeTxReceipt(receipt, tableName, timestamp) {
    const txData = {
        block_number: Number(receipt.blockNumber),
        transaction_hash: receipt.hash,
        transaction_index: Number(receipt.transactionIndex),
        from_address: receipt.from,
        to_address: receipt.to,
        gas_used: Number(receipt.gasUsed),
        gas_price: Number(receipt.effectiveGasPrice ?? receipt.gasPrice),
        logs: JSON.stringify(receipt.logs),
        timestamp: new Date(Number(timestamp) * 1000)
    };

    await insert(tableName, txData);
}

/* ================= 扫描区块 ================= */

async function scanBlocks(contractAddress, startBlock, endBlock, tableName) {
    const currentBlock = await provider.getBlockNumber();
    const targetEnd = endBlock === 'latest' ? currentBlock : endBlock;

    let count = 0;

    for (let blockNumber = startBlock; blockNumber <= targetEnd; blockNumber++) {
        let block;
        try {
            block = await get_block_info(blockNumber);
        } catch {
            continue;
        }

        if (!block || !block.transactions?.length) continue;

        for (const txHash of block.transactions) {
            let receipt;
            try {
                receipt = await get_tx_receipt(txHash);
            } catch {
                continue;
            }

            // 核心过滤：是否出现 token 事件
            if (!receiptHasTokenEvent(receipt, contractAddress)) continue;

            try {
                await storeTxReceipt(receipt, tableName, block.timestamp);
                count++;
            } catch (e) {
                // 重复插入直接跳过
                if (!String(e).includes('duplicate')) {
                    console.error(`Store failed: ${txHash}`, e.message);
                }
            }
        }

        if (blockNumber % 100 === 0) {
            console.log(
                `Scanned block ${blockNumber}, collected ${count} txs`
            );
        }
    }

    return count;
}

/* ================= 主流程 ================= */

async function getTokenTransactions(contractAddress) {
    if (!contractAddress) {
        console.error('Error: contract address required');
        process.exit(1);
    }

    const tableName = await setupTokenTable(contractAddress);
    const { isEmpty, maxBlock } = await getMaxBlockNumber(tableName);

    let startBlock;
    if (isEmpty) {
        console.log('Table empty, locating deployment block...');
        startBlock = await find_deployment_block(contractAddress);
    } else {
        startBlock = Number(maxBlock) + 1;
        console.log(`Resume from block ${startBlock}`);
    }

    const total = await scanBlocks(
        contractAddress,
        startBlock,
        'latest',
        tableName
    );

    console.log(`Done. Stored ${total} new token-related transactions.`);
}

getTokenTransactions(CONTRACT_ADDRESS);
