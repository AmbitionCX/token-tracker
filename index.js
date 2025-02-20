const ethers = require("ethers");
require('dotenv').config()

const provider = new ethers.JsonRpcProvider(process.env.GETH_API)

provider.getBlockNumber().then( res => {
    console.log(res);
})