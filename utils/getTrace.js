import { provider } from './provider.js';
import { get_block_info, get_tx_info, get_tx_receipt, find_deployment_block } from './callBlockInfo.js';
import { getTokenName } from './getTokenName.js';
import { query, insert } from './postgresql.js';

const CONTRACT_ADDRESS = "0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490"; // 示例: 3crv token

// 获取某笔交易的 trace
async function get_tx_trace(txHash) {
    try {
        // 加上 reexec 和 tracer 配置
        const trace = await provider.send("debug_traceTransaction", [
            txHash,
            {
                tracer: "callTracer",   // 可选: "callTracer", "prestateTracer", 或自定义 JS tracer
                reexec: 14000000        // 回溯多少个区块来重建历史状态，数值大一些能避免缺状态错误
            }
        ]);
        return trace;
    } catch (err) {
        console.error(`Error getting trace for ${txHash}:`, err);
        throw err;
    }
}

// 为 trace 建表
async function setupTraceTable(tokenSymbol) {
    const tableName = `token_${tokenSymbol.toLowerCase()}_traces`;
    const createTraceTable = `
        CREATE TABLE IF NOT EXISTS ${tableName} (
            id SERIAL PRIMARY KEY,
            transaction_hash VARCHAR(66) UNIQUE NOT NULL,
            trace JSONB
        )
    `;
    await query(createTraceTable);
    console.log(`Table '${tableName}' for traces verified`);
    return tableName;
}

// 存储 trace
async function storeTxTrace(txHash, traceTable) {
    try {
        const trace = await get_tx_trace(txHash);

        const traceData = {
            transaction_hash: txHash,
            trace: JSON.stringify(trace)
        };

        await insert(traceTable, traceData);
        console.log(`Stored trace for transaction ${txHash}`);
    } catch (error) {
        console.error(`Error storing trace for ${txHash}:`, error);
        throw error;
    }
}

// 扫描区块并抓取 trace
async function scanBlocksForTraces(contractAddress, startBlock, endBlock = 'latest', traceTable) {
    try {
        const currentBlock = await provider.getBlockNumber();
        const targetEndBlock = endBlock === 'latest' ? currentBlock : endBlock;

        let tracesCount = 0;
        for (let blockNumber = startBlock; blockNumber <= targetEndBlock; blockNumber++) {
            try {
                const block = await get_block_info(blockNumber);
                if (block && block.transactions) {
                    const txInfos = await Promise.all(
                        block.transactions.map(tx => get_tx_info(tx))
                    );

                    for (let i = 0; i < block.transactions.length; i++) {
                        const tx = block.transactions[i];
                        const txInfo = txInfos[i];
                        const txFrom = txInfo.from;
                        const txTo = txInfo.to;

                        // 只抓和目标合约相关的交易
                        if (
                            (txFrom && txFrom.toLowerCase() === contractAddress.toLowerCase()) ||
                            (txTo && txTo.toLowerCase() === contractAddress.toLowerCase())
                        ) {
                            await storeTxTrace(tx, traceTable);
                            tracesCount++;
                        }
                    }
                }
            } catch (blockError) {
                console.warn(`Error processing block ${blockNumber}:`, blockError.message);
                continue;
            }

            // 每 100 个区块打印进度
            if (blockNumber % 100 === 0) {
                console.log(`Scanned up to block ${blockNumber}, found ${tracesCount} traces so far`);
            }
        }
        return tracesCount;
    } catch (error) {
        console.error('Error scanning blocks for traces:', error);
        throw error;
    }
}

// 主函数
async function getTokenTraces(contractAddress = CONTRACT_ADDRESS) {
    try {
        const { name: tokenName, symbol: tokenSymbol } = await getTokenName(contractAddress);

        const traceTable = await setupTraceTable(tokenSymbol);

    // const deploymentBlock = await find_deployment_block(contractAddress);
    //   console.log(`Found deployment of ${contractAddress} at block ${deploymentBlock}`);
        const deploymentBlock = 10809467;
        //etherscan的api网络问题调用不了，deploymentBlock先写死，后续再优化
        const tracesCount = await scanBlocksForTraces(
            contractAddress,
            deploymentBlock,
            'latest',
            traceTable
        );

        console.log(`Trace scan finished, ${tracesCount} traces stored`);
    } catch (error) {
        console.error('Error in getTokenTraces:', error);
        throw error;
    }
}

// 运行
getTokenTraces(CONTRACT_ADDRESS);
