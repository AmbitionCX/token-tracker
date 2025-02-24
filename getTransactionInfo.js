const fs = require('fs');
const path = require('path');
const ethers = require("ethers");
require('dotenv').config()

const provider = new ethers.JsonRpcProvider(process.env.GETH_API)

const call_transaction_info = async (transactionHash) => {
    try {
        const txInfo = await provider.getTransactionReceipt(transactionHash);
        
        const destination = './Results';
        if (!fs.existsSync(destination)){ fs.mkdirSync(destination);} // make dir when not exist
        const filePath = path.join(destination, `Tx_${transactionHash}.txt`);
        
        fs.writeFileSync(filePath, JSON.stringify(txInfo.logs, null, 2));

        console.log(`Transaction log saved to ${filePath}`);
    } catch (error) {
        console.error('Error calling transaction:', error);
    }
};

call_transaction_info("0x58d99b5a552d9a90b23fe3db1a4e942104da02fa5e9fafbae03f1d6a577b98ed")
