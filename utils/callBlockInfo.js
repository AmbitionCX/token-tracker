import fs from 'fs';
import path from 'path';
import axios from 'axios';
import { provider } from './provider.js';

export const resultStore = async (data, customFileName = null) => {
    try {
        const destination = './Results';
        if (!fs.existsSync(destination)) {
            fs.mkdirSync(destination, { recursive: true });
        }

        const fileName = customFileName || 'data.json';
        const filePath = path.join(destination, fileName);

        fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
        return filePath;
    } catch (error) {
        console.error('Error storing data:', error);
        throw error;
    }
};

export const get_block_info = async (blockNumber) => {
    try {
        const blockInfo = await provider.getBlock(blockNumber, true);
        return blockInfo;
    } catch (error) {
        console.error('Error calling block:', error);
    }
};

export const store_block_info = async (blockNumber) => {
    try {
        const blockInfo = await get_block_info(blockNumber);
        const blockFileName = `Block_${blockNumber}.txt`;
        const blockFilePath = await resultStore(blockInfo, blockFileName);

        console.log(`Block saved to ${blockFilePath}`);
    } catch (error) {
        console.error('Error storing data:', error);
        throw error;
    }
}

export const get_tx_info = async (txHash) => {
    try {
        const txInfo = await provider.getTransaction(txHash);
        return txInfo;
    } catch (error) {
        console.error('Error tracing transaction:', error);
    }
}

export const store_tx_info = async (txHash) => {
    try {
        const txInfo = await get_tx_info(txHash);
        const txFileName = `Tx_${txHash}.txt`;
        const txFilePath = await resultStore(txInfo, txFileName);

        console.log(`Transaction saved to ${txFilePath}`);
    } catch (error) {
        console.error('Error tracing transaction:', error);
    }
}

export const get_tx_receipt = async (txHash) => {
    try {
        const receiptInfo = await provider.getTransactionReceipt(txHash);
        return receiptInfo;
    } catch (error) {
        console.error('Error tracing transaction:', error);
    }
}

export const store_tx_receipt = async (txHash) => {
    try {
        const receiptInfo = await get_tx_receipt(txHash);
        const receiptFileName = `Receipt_${txHash}.txt`;
        const receiptFilePath = await resultStore(receiptInfo, receiptFileName);

        console.log(`Receipt saved to ${receiptFilePath}`);
    } catch (error) {
        console.error('Error tracing transaction:', error);
    }
}

export const find_deployment_block = async (contractAddress) => {
    try {
        const apiKey = process.env.ETHERSCAN_API_KEY;
        const base_url = process.env.ETHERSCAN_API_URL;

        const etherscan_api_url = `${base_url}&module=contract&action=getcontractcreation&contractaddresses=${contractAddress}&apikey=${apiKey}`;
        const response = await axios.get(etherscan_api_url, { timeout: 15000 });

        // response structure
        // [
        //   {
        //     contractAddress: '',
        //     contractCreator: '',
        //     txHash: '',
        //     blockNumber: '',
        //     timestamp: '',
        //     contractFactory: '',
        //     creationBytecode: ''
        //   }
        // ]

        const block_number = response.data.result[0].blockNumber;
        return Number(block_number);
    } catch (error) {
        console.log(error);
        throw error;
    }
}
