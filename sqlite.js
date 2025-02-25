const sqlite3 = require('sqlite3').verbose();

// Create a connection to the database (or create it if it doesn't exist)
const db = new sqlite3.Database('./Sqlite/TransactionInfo.sqlite', (err) => {
    if (err) {
        console.error('Error opening database', err.message);
    } else {
        console.log('Connected to the SQLite database.');
    }
})

function createTransactionInfoTable(db) {
    // Create the CRV transaction table
    db.run(`CREATE TABLE IF NOT EXISTS Transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        blockNumber INTEGER,
        contractAddress TEXT,
        cumulativeGasUsed TEXT,
        fromAddress TEXT,
        toAddress TEXT,
        gasUsed TEXT,
        gasPrice TEXT,
        blobGasUsed TEXT,
        blobGasPrice TEXT,
        transactionHash TEXT UNIQUE,
        transactionIndex INTEGER,
        status INTEGER
        
    );`)

    db.run(`CREATE TABLE IF NOT EXISTS TransactionLogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transactionHash TEXT,
        logIndex INTEGER,
        address TEXT,
        blockHash TEXT,
        blockNumber INTEGER,
        data TEXT,
        topics TEXT,
        FOREIGN KEY (transactionHash) REFERENCES Transactions(transactionHash)
    );`);
};

// Close the database connection
function closeDatabase(db) {
    db.close((err) => {
        if (err) {
            console.error('Error closing database', err.message);
        } else {
            console.log('Database connection closed.');
        }
    });
}

module.exports = { db, createTransactionInfoTable, closeDatabase };



