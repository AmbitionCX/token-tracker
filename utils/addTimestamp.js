import { get_block_info } from './callBlockInfo.js'
import { query } from './postgresql.js';

const tableName = process.argv[2] || null;

async function addMissingTimestamps(tableName) {
    try {
        // Get all transactions without timestamps
        const result = await query(
            `SELECT transaction_hash, block_number FROM ${tableName} WHERE timestamp IS NULL`
        );

        if (result.rows.length === 0) {
            console.log('No missing timestamps found');
            return;
        }

        console.log(`Found ${result.rows.length} transactions without timestamps`);

        for (const row of result.rows) {
            try {
                // Get block info to extract timestamp
                const block = await get_block_info(Number(row.block_number));

                if (block && block.timestamp) {
                    // Update the timestamp in database
                    await query(
                        `UPDATE ${tableName} SET timestamp = $1 WHERE transaction_hash = $2`,
                        [new Date(Number(block.timestamp) * 1000), row.transaction_hash]
                    );
                    console.log(`Updated timestamp for tx ${row.transaction_hash}`);
                }
            } catch (error) {
                console.warn(`Error updating timestamp for tx ${row.transaction_hash}:`, error.message);
            }
        }
        console.log('Timestamp update completed');
    } catch (error) {
        console.error('Error in addMissingTimestamps:', error);
    }
}

await addMissingTimestamps(tableName);