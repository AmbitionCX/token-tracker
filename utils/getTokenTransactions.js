import { provider } from './provider.js';
import { get_block_info, get_tx_info, get_tx_receipt, find_deployment_block } from './callBlockInfo.js'
import { getTokenName } from './getTokenName.js'
import { query, insert } from './postgresql.js';

const CONTRACT_ADDRESS = "0x514910771AF9Ca656af840dff83E8264EcF986CA"

// Get token info and create transaction table
async function setupTokenTable(contractAddress) {
    try {
        // Get token name and symbol
        const { name: tokenName, symbol: tokenSymbol } = await getTokenName(contractAddress);
        console.log(`Token: ${tokenName} (${tokenSymbol})`);
        const tableName = `${tokenSymbol.toLowerCase()}_transactions`;

        // Create table for this token
        const createTransactionTable = `
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
                logs JSONB
            )`;

        await query(createTransactionTable);
        console.log(`Table '${tableName}' verified`);

        return tableName;
    } catch (error) {
        console.log('Error setting up token table:', error);
        throw error;
    }
}

// Store transaction receipt     
async function storeTxReceipt(txHash, contractAddress, tableName) {
    try {
        const receipt = await get_tx_receipt(txHash);

        // Prepare data for insertion           
        const txData = {
            block_number: Number(receipt.blockNumber),
            transaction_hash: receipt.hash.toString(),
            transaction_index: Number(receipt.index),
            from_address: receipt.from.toString(),
            to_address: receipt.to.toString(),
            contract_address: contractAddress,
            gas_used: Number(receipt.gasUsed),
            cumulative_gas_used: Number(receipt.cumulativeGasUsed),
            gas_price: Number(receipt.gasPrice),
            logs: JSON.stringify(receipt.logs)
        };

        // Insert into PostgreSQL               
        const result = await insert(tableName, txData);
        console.log(`Stored transaction ${txHash} in block ${receipt.blockNumber}`);

        // return result;
    } catch (error) {
        console.error(`Error processing transaction ${txHash}:`, error);
        throw error;
    }
}

// Scan blocks
async function scanBlocks(contractAddress, startBlock, endBlock = 'latest', tableName) {
    try {
        const currentBlock = await provider.getBlockNumber();
        const targetEndBlock = endBlock === 'latest' ? currentBlock : endBlock;

        let transactionsCount = 0;
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

                        if (
                            (txFrom && txFrom.toLowerCase() === contractAddress.toLowerCase()) ||
                            (txTo && txTo.toLowerCase() === contractAddress.toLowerCase())
                        ) {
                            await storeTxReceipt(tx, contractAddress, tableName);
                            transactionsCount++;
                        }
                    }
                }
            } catch (blockError) {
                console.warn(`Error processing block ${blockNumber}:`, blockError.message);
                continue;
            }

            // Progress update every 100 blocks
            if (blockNumber % 100 === 0) {
                console.log(`Scanned up to block ${blockNumber}, found ${transactionsCount} transactions so far`);
            }
        }
    } catch (error) {
        console.error('Error scanning blocks:', error);
        throw error;
    }
    return transactionsCount;
}

async function getTokenTransactions(contractAddress = CONTRACT_ADDRESS) {
    try {
        // Setup token table            
        const tableName = await setupTokenTable(contractAddress);

        // Find deployment block        
        const deploymentBlock = await find_deployment_block(contractAddress);

        // Scan until latest block 
        const transactionsCount = await scanBlocks(
            contractAddress,
            deploymentBlock,
            'latest', // Scan to latest block
            tableName
        );

        console.log(`Scan finished, ${transactionsCount} Transactions found`);
    } catch (error) {
        console.error('Error in getTokenTransactions:', error);
        throw error;
    }
}

getTokenTransactions(CONTRACT_ADDRESS);