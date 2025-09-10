import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const { Pool } = pg;

// Create connection pool
const pool = new Pool({
    host: process.env.POSTGRESQL_HOST,
    port: process.env.DPOSTGRESQL_PORT,
    database: process.env.POSTGRESQL_DATABASE,
    user: process.env.POSTGRESQL_USER,
    password: process.env.POSTGRESQL_PASSWORD,

    // Optional: connection pool settings
    max: 20, // maximum number of clients in the pool
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
});

// Generic query function
export async function query(text, params) {
    const client = await pool.connect();
    try {
        const result = await client.query(text, params);
        return result;
    } finally {
        client.release();
    }
}

// Check database connection
export async function checkConnection() {
    try {
        await query('SELECT NOW()');
        console.log('PostgreSQL connection successful');
        return true;
    } catch (error) {
        console.error('PostgreSQL connection failed:', error.message);
        return false;
    }
}
checkConnection();

// Insert with returning ID
export async function insert(table, data) {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const placeholders = keys.map((_, i) => `$${i + 1}`).join(', ');
    const columns = keys.join(', ');

    const text = `INSERT INTO ${table} (${columns}) VALUES (${placeholders}) RETURNING *`;
    const result = await query(text, values);
    return result.rows[0];
}

// Update by ID
export async function update(table, id, data) {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const setClause = keys.map((key, i) => `${key} = $${i + 1}`).join(', ');

    const text = `UPDATE ${table} SET ${setClause} WHERE id = $${keys.length + 1} RETURNING *`;
    const result = await query(text, [...values, id]);
    return result.rows[0];
}

// Delete by ID
export async function remove(table, id) {
    const text = `DELETE FROM ${table} WHERE id = $1 RETURNING *`;
    const result = await query(text, [id]);
    return result.rows[0];
}

// Close pool (call when shutting down)
export async function closePool() {
    await pool.end();
    console.log('PostgreSQL pool closed');
}

// check if a table is empty, and get the latest block number
export async function getMaxBlockNumber(tableName) {
    const maxBlockNumber = `SELECT MAX(block_number) AS latest_block FROM ${tableName}`;
    const tableResult = await query(maxBlockNumber);
    const maxBlock = tableResult.rows[0].latest_block;

    if (maxBlock === null) {
        return {
            isEmpty: true,
            maxBlock: null
        };
    } else {
        return {
            isEmpty: false,
            maxBlock: maxBlock
        };
    }
}

export default {
    query,
    checkConnection,
    closePool,
    insert,
    update,
    remove,
    getMaxBlockNumber
};