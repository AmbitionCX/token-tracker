require('dotenv').config();
const sqlite3 = require('sqlite3').verbose();
const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// 通过环境变量获取 Geth API 地址
const provider = new ethers.JsonRpcProvider(process.env.GETH_API);

// ERC-20 合约的 Transfer 事件 ABI
const contractABI = [
  "event Transfer(address indexed from, address indexed to, uint256 value)"
];

// 创建 ERC-20 合约实例
const tokenAddress = '0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490';  // 代币合约地址
const tokenContract = new ethers.Contract(tokenAddress, contractABI, provider);

// 创建 SQLite 数据库目录
const dbDirectory = './Sqlite';  // 数据库存放的目录
const dbFile = path.join(dbDirectory, 'TransferInfo.db');

// 如果目录不存在，则创建目录
if (!fs.existsSync(dbDirectory)) {
  fs.mkdirSync(dbDirectory, { recursive: true });
  console.log(`Directory ${dbDirectory} created.`);
}

// 创建新的 SQLite 数据库（如果没有则创建）
const db = new sqlite3.Database(dbFile, (err) => {
  if (err) {
    return console.error('Error creating database:', err.message);
  }
  console.log('Database created successfully at:', dbFile);
});

// 创建表格
const createTransactionInfoTable = () => {
  const createTableQuery = `
    CREATE TABLE IF NOT EXISTS Transactions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      blockNumber INTEGER,
      contractAddress TEXT,
      cumulativeGasUsed INTEGER,
      fromAddress TEXT,
      toAddress TEXT,
      gasUsed INTEGER,
      gasPrice INTEGER,
      blobGasUsed INTEGER,
      blobGasPrice INTEGER,
      transactionHash TEXT,
      transactionIndex INTEGER,
      status INTEGER
    );
  `;

  db.run(createTableQuery, (err) => {
    if (err) {
      return console.error('Error creating table:', err.message);
    }
    console.log('Transactions table created successfully');
  });
};

// 插入交易数据到数据库
const insertTransactionData = (txInfo) => {
  const insertTransactionQuery = `
    INSERT INTO Transactions (
      blockNumber, contractAddress, cumulativeGasUsed, fromAddress, toAddress,
      gasUsed, gasPrice, blobGasUsed, blobGasPrice, transactionHash, transactionIndex, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
  `;

  db.run(insertTransactionQuery, [
    txInfo.blockNumber,
    txInfo.contractAddress,
    txInfo.cumulativeGasUsed,
    txInfo.from,
    txInfo.to,
    txInfo.gasUsed,
    txInfo.gasPrice,
    txInfo.blobGasUsed,
    txInfo.blobGasPrice,
    txInfo.hash,
    txInfo.index,
    txInfo.status
  ], function (err) {
    if (err) {
      return console.error('Error inserting transaction:', err.message);
    }
    console.log(`Transaction inserted with ID: ${this.lastID}`);
  });
};

// 获取交易哈希并保存交易信息到数据库
const getTransactionHashesAndSave = async () => {
  try {
    const latestBlock = await provider.getBlockNumber();  // 获取最新区块号
    const fromBlock = latestBlock - 10000;  // 查询的起始区块号（向前推10,000个区块）

    // 查询 Transfer 事件
    const events = await tokenContract.queryFilter(
      tokenContract.filters.Transfer(),
      fromBlock,
      latestBlock
    );

    console.log(`Found ${events.length} transactions:`);

    events.forEach(async (event) => {
      console.log(`Transaction Hash: ${event.transactionHash}`);
      // await callTransactionInfo(event.transactionHash);  // 使用交易哈希查询交易信息
    });

  } catch (error) {
    console.error('Error fetching transactions:', error);
  }
};

// 查询交易详细信息并保存到数据库
const callTransactionInfo = async (transactionHash) => {
  try {
    const txInfo = await provider.getTransactionReceipt(transactionHash);

    // 将交易数据插入数据库
    insertTransactionData(txInfo);

  } catch (error) {
    console.error('Error calling transaction:', error);
  }
};

// 执行函数查询并保存交易信息
createTransactionInfoTable();  // 创建数据库表格
getTransactionHashesAndSave(); // 获取交易哈希并保存信息