require('dotenv').config();
const { ethers } = require('ethers');

// 通过环境变量获取 Geth API 地址
const provider = new ethers.JsonRpcProvider(process.env.GETH_API);

const tokenAddress = '0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490';  // 代币合约地址
const numberOfTransactions = 1000;  // 指定要查询的交易数量

// ERC-20 合约的 Transfer 事件 ABI
const contractABI = [
  "event Transfer(address indexed from, address indexed to, uint256 value)"
];

// 创建 ERC-20 合约实例
const tokenContract = new ethers.Contract(tokenAddress, contractABI, provider);

// 获取交易哈希
async function getTransactionHashes() {
  try {
    const latestBlock = await provider.getBlockNumber();  // 获取最新区块号
    const fromBlock = latestBlock - 100000;  // 查询的起始区块号（向前推10个区块）

    // 查询 Transfer 事件
    const events = await tokenContract.queryFilter(
      tokenContract.filters.Transfer(), 
      fromBlock, 
      latestBlock
    );

    // 输出交易哈希
    console.log(`Found ${events.length} transactions:`);
    events.forEach((event, index) => {
      console.log(`${index + 1}. Transaction Hash: ${event.transactionHash}`);
    });
  } catch (error) {
    console.error('Error fetching transactions:', error);
  }
}

getTransactionHashes();