import { provider } from './provider.js';
import { keccak256 } from "js-sha3";
import dotenv from 'dotenv';
dotenv.config();

const contractAddress = "0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490";

// 辅助函数：将hex转换为大整数
function hexToBigInt(hex) {
    return BigInt(hex.startsWith('0x') ? hex : '0x' + hex);
}

// 辅助函数：将数字转换为hex
function numberToHex(num) {
    return '0x' + num.toString(16);
}

// 将输入数据按32字节(64位hex)分割
function splitInputData(inputData) {
    const chunks = [];
    for (let i = 0; i < inputData.length; i += 64) {
        chunks.push(inputData.slice(i, i + 64));
    }
    return chunks;
}

// 解码参数
function decodeParameter(type, value) {
    if (type === 'address') {
        return '0x' + value.slice(24); // 取后20字节
    } else if (type === 'uint256') {
        return hexToBigInt('0x' + value);
    }
    return value;
}

async function getLatestBlock() {
    const response = await axios.post(GETH_API, {
        jsonrpc: "2.0",
        id: 1,
        method: "eth_blockNumber",
        params: []
    });
    return hexToBigInt(response.data.result);
}

async function getBlockTransactions(blockNum) {
    try {
        const response = await axios.post(GETH_API, {
            jsonrpc: "2.0",
            id: 1,
            method: "eth_getBlockByNumber",
            params: [numberToHex(blockNum), true]
        });
        return response.data.result?.transactions?.filter(tx => 
            tx.to?.toLowerCase() === contractAddress.toLowerCase()
        ) || [];
    } catch (error) {
        console.error(`Error fetching block ${blockNum}:`, error);
        return [];
    }
}

async function getTransactions(fromBlock, toBlock) {
    console.log(`Fetching transactions from block ${fromBlock} to ${toBlock}...`);
    const blockNumbers = Array.from(
        { length: Number(toBlock - fromBlock + BigInt(1)) }, 
        (_, i) => fromBlock + BigInt(i)
    );
    const transactions = await Promise.all(blockNumbers.map(getBlockTransactions));
    return transactions.flat();
}

async function parseTransactions(transactions) {
    const selectors = {
        transfer: "0x" + keccak256("transfer(address,uint256)").slice(0, 8),
        transferFrom: "0x" + keccak256("transferFrom(address,address,uint256)").slice(0, 8),
        approve: "0x" + keccak256("approve(address,uint256)").slice(0, 8)
    };

    const parsedTxs = [];

    for (const tx of transactions) {
        if (!tx.input || tx.input === "0x") continue;

        const methodId = tx.input.slice(0, 10);
        const inputData = tx.input.slice(10);
        const params = splitInputData(inputData);

        try {
            if (methodId === selectors.transfer) {
                parsedTxs.push({ 
                    function: "transfer", 
                    hash: tx.hash, 
                    from: tx.from, 
                    to: decodeParameter('address', params[0]), 
                    value: decodeParameter('uint256', params[1]).toString()
                });

            } else if (methodId === selectors.transferFrom) {
                parsedTxs.push({ 
                    function: "transferFrom", 
                    hash: tx.hash, 
                    from: decodeParameter('address', params[0]), 
                    to: decodeParameter('address', params[1]), 
                    value: decodeParameter('uint256', params[2]).toString(), 
                    caller: tx.from 
                });

            } else if (methodId === selectors.approve) {
                parsedTxs.push({ 
                    function: "approve", 
                    hash: tx.hash, 
                    from: tx.from, 
                    spender: decodeParameter('address', params[0]), 
                    value: decodeParameter('uint256', params[1]).toString()
                });
            }
        } catch (error) {
            console.error(`Failed to decode tx ${tx.hash}:`, error.message);
        }
    }

    console.log("\nParsed Transactions:");
    parsedTxs.forEach(tx => {
        console.log('\nTransaction:', {
            ...tx,
            hash: tx.hash,
            function: tx.function,
            value: tx.value === '115792089237316195423570985008687907853269984665640564039457584007913129639935' 
                ? 'MAX_UINT256' 
                : tx.value
        });
    });
    
    return parsedTxs;
}

async function main() {
    try {
        const latestBlock = await getLatestBlock();
        const fromBlock = latestBlock - BigInt(3000);
        const transactions = await getTransactions(fromBlock, latestBlock);

        if (transactions.length > 0) {
            console.log(`Found ${transactions.length} transactions interacting with the contract.`);
            await parseTransactions(transactions);
        } else {
            console.log("No transactions found for this contract in the last 3000 blocks.");
        }
    } catch (error) {
        console.error("Error in main function:", error);
    }
}

main().catch(console.error);