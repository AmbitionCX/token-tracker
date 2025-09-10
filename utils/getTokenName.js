import { ethers } from "ethers";
import { provider } from './provider.js';

export const getTokenName = async (tokenAddress) => {
    // ERC20 ABI
    const erc20Abi_string = [
        "function name() external view returns (string)",
        "function symbol() external view returns (string)",
    ];
    const tokenContract_string = new ethers.Contract(tokenAddress, erc20Abi_string, provider);

    const erc20Abi_byte32 = [
        "function name() external view returns (bytes32)",
        "function symbol() external view returns (bytes32)",
    ];
    const tokenContract_byte32 = new ethers.Contract(tokenAddress, erc20Abi_byte32, provider);

    try {
        let name, symbol;

        try {
            name = await tokenContract_string.name();
            symbol = await tokenContract_string.symbol();
        } catch (err) {
            console.warn("Failed to fetch name as string", err);
            name = null;
            symbol = null;
        }

        if (!name || !symbol) {

            const nameBytes = await tokenContract_byte32.name();
            const symbolBytes = await tokenContract_byte32.symbol();
            name = ethers.decodeBytes32String(nameBytes);
            symbol = ethers.decodeBytes32String(symbolBytes);
        }

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
