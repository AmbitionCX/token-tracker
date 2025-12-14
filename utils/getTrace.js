import { provider } from './provider.js';
import { getTokenName } from './getTokenName.js';
import { query, insert } from './postgresql.js';

const CONTRACT_ADDRESS = process.argv[2] || null;

async function get_tx_trace(txHash) {
    try {
        const trace = await provider.send("debug_traceTransaction", [
            txHash,
            {
                tracer: "callTracer",
                reexec: 14000000
            }
        ]);
        return trace;
    } catch (err) {
        console.error(`Error getting trace for ${txHash}:`, err);
        throw err;
    }
}

async function setupTraceTable(tokenSymbol) {
    const tableName = `${tokenSymbol.toLowerCase()}_traces`;
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

async function getLastTracedTxId(traceTable) {
    try {
        const queryText = `
            SELECT MAX(id) as last_id
            FROM ${traceTable}
        `;               
        const result = await query(queryText);
        return result.rows[0].last_id || 0;
    } catch (error) {
        console.error('Error getting last traced tx ID:', error);
        return 0;
    }   
} 

async function getNextTransaction(tokenSymbol, lastProcessedId) {
    const transactionsTable = `${tokenSymbol.toLowerCase()}_transactions`;

    const getNextTxQuery = `
        SELECT id, transaction_hash
        FROM ${transactionsTable}
        WHERE id > $1
        ORDER BY id ASC
        LIMIT 1
    `;

    const result = await query(getNextTxQuery, [lastProcessedId]);

    if (result.rows.length === 0) {
        return null;
    }

    return {
        id: result.rows[0].id,
        transaction_hash: result.rows[0].transaction_hash
    };
}

async function storeTxTrace(txHash, traceTable) {
    try {
        const trace = await get_tx_trace(txHash);

        const traceData = {
            transaction_hash: txHash,
            trace: JSON.stringify(trace)
        };

        await insert(traceTable, traceData);
        console.log(`Stored trace for ${txHash}`);
    } catch (error) {
        console.error(`Error storing trace for ${txHash}:`, error);
        throw error;
    }
}

async function fetchTraces(contractAddress, traceTable) {     
    try {                
        const { symbol: tokenSymbol } = await getTokenName(contractAddress);
        
        // Get the last processed transaction ID from trace table           
        const lastProcessedId = await getLastTracedTxId(traceTable);             
        console.log(`Latest traced transaction: ${lastProcessedId}`);               
        
        let processedCount = 0;           
        let currentId = lastProcessedId;  
        
        while (true) {   
            // Get next transaction to process             
            const nextTransaction = await getNextTransaction(tokenSymbol, currentId);                
        
            if (!nextTransaction) {       
                console.log('No more transactions to process.');            
                break;   
            }            
        
            const { id: transactionId, transaction_hash: txHash } = nextTransaction;
            console.log(`Processing ID: ${transactionId}, Hash: ${txHash}`);         
        
            try {        
                // Fetch and store trace  
                await storeTxTrace(txHash, traceTable);      
                currentId = transactionId;
                processedCount++;         
        
                // Add a small delay to avoid rate limiting
                await new Promise(resolve => setTimeout(resolve, 200));     
        
            } catch (error) {             
                console.error(`Failed to process transaction ID ${transactionId} (${txHash}):`, error.message);   
                // Optional: retry logic              
                currentId = transactionId;
                continue;
            } 
        }                
        return processedCount;            
        
    } catch (error) {    
        console.error('Error processing traces sequentially:', error);      
        throw error;     
    }   
}

// Main function
async function getTokenTraces(contractAddress = CONTRACT_ADDRESS) {
    try {
        const { name: tokenName, symbol: tokenSymbol } = await getTokenName(contractAddress);
        const traceTable = await setupTraceTable(tokenSymbol);

        const tracesCount = await fetchTraces(
            contractAddress,
            traceTable
        );

        console.log(`Trace scan finished, ${tracesCount} traces stored`);
    } catch (error) {
        console.error('Error in getTokenTraces:', error);
        throw error;
    }
}

getTokenTraces(CONTRACT_ADDRESS);
