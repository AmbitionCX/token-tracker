const ethers = require("ethers");
require('dotenv').config();

// 打印加载的 URL，确认 `.env` 文件是否正确加载
console.log('Geth Node URL:', process.env.GETH_API);

// 初始化 provider
const provider = new ethers.JsonRpcProvider(process.env.GETH_API);

// 测试节点连接
const testConnection = async () => {
    try {
        // 获取最新区块号
        const blockNumber = await provider.getBlockNumber();
        console.log('Connection successful. Latest block number:', blockNumber);

        // 检查节点同步状态
        const syncStatus = await provider.send('eth_syncing', []);
        console.log('Sync status:', syncStatus);
    } catch (error) {
        console.error('Connection failed:', error.message || error);
        console.log(error);  // 打印更多错误信息
    }
};

// 检查指定区块是否存在
const checkBlockExists = async (blockNumber) => {
    try {
        const blockInfo = await provider.getBlock(blockNumber, true);
        if (!blockInfo) {
            console.error(`Block ${blockNumber} not found`);
        } else {
            console.log(`Block ${blockNumber} exists. Block info:`, blockInfo);
        }
    } catch (error) {
        console.error('Error checking block:', error.message || error);
        console.log(error);  // 打印更多错误信息
    }
};

// 运行调试
const debug = async () => {
    await testConnection();
    await checkBlockExists(10809467); // 替换为你要检查的区块号
};

// 执行调试
debug();