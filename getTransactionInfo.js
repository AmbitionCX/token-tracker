const fs = require('fs');
const path = require('path');
const ethers = require("ethers");
require('dotenv').config()

const { db, createTransactionInfoTable, closeDatabase } = require('./sqlite.js');

const provider = new ethers.JsonRpcProvider(process.env.GETH_API)

const callTransactionInfo = async (transactionHash) => {
    // Remove TransactionInfo.sqlite file if you need to start from empty
    createTransactionInfoTable(db);

    try {
        const txInfo = await provider.getTransactionReceipt(transactionHash);

        const destination = './Results';
        if (!fs.existsSync(destination)) { fs.mkdirSync(destination); } // make dir when not exist
        const filePath = path.join(destination, `Tx_${transactionHash}.txt`);

        fs.writeFileSync(filePath, JSON.stringify(txInfo, null, 2));

        console.log(`Transaction log saved to ${filePath}`);

        const insertTransaction = `INSERT INTO Transactions (
            blockNumber, contractAddress, cumulativeGasUsed, fromAddress, toAddress,
            gasUsed, gasPrice, blobGasUsed, blobGasPrice, transactionHash, transactionIndex, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;

        db.run(insertTransaction, [
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

            // Insert into TransactionLogs table
            const insertLog = `INSERT INTO TransactionLogs (
                transactionHash, logIndex, address, blockHash, blockNumber, data, topics
            ) VALUES (?, ?, ?, ?, ?, ?, ?)`;

            txInfo.logs.forEach(log => {
                db.run(insertLog, [
                    log.transactionHash,
                    log.index,
                    log.address,
                    log.blockHash,
                    log.blockNumber,
                    log.data,
                    JSON.stringify(log.topics) // Convert topics array to JSON string
                ], function (err) {
                    if (err) {
                        return console.error('Error inserting log:', err.message);
                    }
                });
            });
        });
        closeDatabase(db);
    } catch (error) {
        console.error('Error calling transaction:', error);
    }
};

callTransactionInfo("0x58d99b5a552d9a90b23fe3db1a4e942104da02fa5e9fafbae03f1d6a577b98ed")
