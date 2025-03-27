const axios = require('axios'); 
const { Web3 } = require('web3');
const web3 = new Web3("http://10.222.117.105:8545");

// 目标合约地址和ABI
const contractAddress = "0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490";
const contractABI = [
    { "constant": false, "inputs": [{"name": "_to", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "transfer", "outputs": [{"name": "", "type": "bool"}], "type": "function" },
    { "constant": false, "inputs": [{"name": "_from", "type": "address"}, {"name": "_to", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "transferFrom", "outputs": [{"name": "", "type": "bool"}], "type": "function" },
    { "constant": false, "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function" }
];

const contract = new web3.eth.Contract(contractABI, contractAddress);

async function getLatestBlock() {
    const latestBlockNumber = await web3.eth.getBlockNumber();
    return BigInt(latestBlockNumber); // 转换为 BigInt 类型
}

// 获取单个区块的交易数据
async function getBlockTransactions(blockNum) {
    try {
        const block = await web3.eth.getBlock(blockNum, true);
        return block?.transactions?.filter(tx => tx.to?.toLowerCase() === contractAddress.toLowerCase()) || [];
    } catch (error) {
        console.error(`Error fetching block ${blockNum}:`, error);
        return [];
    }
}

// 批量获取区块数据
async function getTransactions(fromBlock, toBlock) {
    console.log(`Fetching transactions from block ${fromBlock} to ${toBlock}...`);
    const blockNumbers = Array.from({ length: Number(toBlock - fromBlock + BigInt(1)) }, (_, i) => fromBlock + BigInt(i));
    const transactions = await Promise.all(blockNumbers.map(getBlockTransactions));
    return transactions.flat();
}

// 解析交易并提取函数调用
async function parseTransactions(transactions) {
    const selectors = {
        transfer: web3.eth.abi.encodeFunctionSignature("transfer(address,uint256)"),
        transferFrom: web3.eth.abi.encodeFunctionSignature("transferFrom(address,address,uint256)"),
        approve: web3.eth.abi.encodeFunctionSignature("approve(address,uint256)")
    };

    const parsedTxs = [];

    for (const tx of transactions) {
        if (!tx.input || tx.input === '0x') continue;

        const methodId = tx.input.slice(0, 10);
        let decodedParams;

        try {
            if (methodId === selectors.transfer) {
                decodedParams = web3.eth.abi.decodeParameters(["address", "uint256"], tx.input.slice(10));
                parsedTxs.push({ function: "transfer", hash: tx.hash, from: tx.from, to: decodedParams[0], value: decodedParams[1] });
            } else if (methodId === selectors.transferFrom) {
                decodedParams = web3.eth.abi.decodeParameters(["address", "address", "uint256"], tx.input.slice(10));
                parsedTxs.push({ function: "transferFrom", hash: tx.hash, from: decodedParams[0], to: decodedParams[1], value: decodedParams[2], caller: tx.from });
            } else if (methodId === selectors.approve) {
                decodedParams = web3.eth.abi.decodeParameters(["address", "uint256"], tx.input.slice(10));
                parsedTxs.push({ function: "approve", hash: tx.hash, from: tx.from, spender: decodedParams[0], value: decodedParams[1] });
            }
        } catch (error) {
            console.error(`Failed to decode tx ${tx.hash}:`, error.message);
        }
    }

    console.log("Parsed Transactions:", parsedTxs);
    return parsedTxs;
}

// 主函数
async function main() {
    try {
        const latestBlock = await getLatestBlock();
        const fromBlock = latestBlock - BigInt(3000); // 获取最近3000个区块
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
