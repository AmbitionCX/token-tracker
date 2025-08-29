import { ethers } from "ethers";
import { provider } from './provider.js';

export const getTokenName = async (tokenAddress) => {
    // ERC20 ABI (minimal for name/symbol)
    const erc20Abi = [
        "function name() external view returns (string)",
        "function symbol() external view returns (string)",
    ];
    const tokenContract = new ethers.Contract(tokenAddress, erc20Abi, provider);

    try {
        const name = await tokenContract.name();
        const symbol = await tokenContract.symbol();
        // console.log(`Token ${tokenAddress} Name: ${name}, Symbol: ${symbol}`);

        return { name, symbol };
    } catch (error) {
        console.error('ERC20 token error:', error);
        throw error;
    }
}

export const getSwapPair = async (swapAddress) => {

    // Contract address and ABI (minimal ABI for the public variables)
    const contractAddress = swapAddress;
    const abi = [
        "function token0() external view returns (address)",
        "function token1() external view returns (address)",
    ];
    // Create the contract instance
    const contract = new ethers.Contract(contractAddress, abi, provider);

    try {
        const token0 = await contract.token0();
        const token1 = await contract.token1();
        getTokenName(token0);
        getTokenName(token1);

    } catch (error) {
        console.error('Swap pair error:', error);
        throw error;
    }
};
