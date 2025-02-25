const fs = require('fs');
const path = require('path');
const ethers = require("ethers");
require('dotenv').config()

const provider = new ethers.JsonRpcProvider(process.env.GETH_API)

const call_block_info = async (blockNumber) => {
    try {
        const blockInfo = await provider.getBlock(blockNumber, true);
        
        const destination = './Results';
        if (!fs.existsSync(destination)){ fs.mkdirSync(destination);} // make dir when not exist
        const filePath = path.join(destination, `Block_${blockNumber}.txt`);
        
        fs.writeFileSync(filePath, JSON.stringify(blockInfo, null, 2));

        console.log(`BlockInfo saved to ${filePath}`);
    } catch (error) {
        console.error('Error tracing transaction:', error);
    }
};

// The token is deployed in block 10809467
// call_block_info(10809467)
call_block_info("latest")