import { provider } from './provider.js';

// 3Crv Token Transfer Event
// Source code: https://etherscan.io/token/0x6c3f90f043a72fa612cbac8115ee7e52bde6e490#code

// Keccak256(Transfer(address,address,uint256)) = 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
const EVENT_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef" // Topic[0]

// Keccak256(Approval(address,address,uint256)) = 0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925
const EVENT_APPROVAL = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925" // Topic[0]

// transfer(address,uint256)
const DATA_TRANSFER = "0xa9059cbb" // First 10 of tx.data string

// transferFrom(address,address,uint256)
const DATA_TRANSFERFROM = "0x23b872dd" // First 10 of tx.data string

// approve(address,uint256)
const DATA_APPROVE = "0x095ea7b3" // First 10 of tx.data string

export const parseTransferTx = async (txHash) => {
    try {
        const tx = await provider.getTransaction(txHash)
        const receipt = await provider.getTransactionReceipt(txHash)

        if (!tx || !receipt) {
            throw new Error("Transaction not found")
        }

        const transfers = []
        const txData = tx.data.toLowerCase()

        // Parse Transfer function
        if (txData.startsWith(DATA_TRANSFER)) {
            const rawData = txData.slice(DATA_TRANSFER.length)
            
            if (rawData.length === 128) {
                const toHex = '0x' + rawData.slice(0, 64).replace(/^0+/, '') // remove leading 0
                const valueHex = '0x' + rawData.slice(64, 128).replace(/^0+/, '')
                const valueDex = Number(valueHex).toString()

                transfers.push({
                    contract: tx.to,   // Contract Address
                    from: tx.from,     // Caller
                    to: toHex,
                    value: valueDex,
                    type: "transfer"
                })
            }
        }

        // Parse TransferFrom function
        if (txData.startsWith(DATA_TRANSFERFROM)) {
            const rawData = txData.slice(DATA_TRANSFERFROM.length)
            
            if (rawData.length === 192) {
                const fromHex = '0x' + rawData.slice(0, 64).replace(/^0+/, '') // remove leading 0
                const toHex = '0x' + rawData.slice(64, 128).replace(/^0+/, '')
                const valueHex = '0x' + rawData.slice(128, 192).replace(/^0+/, '')
                const valueDex = Number(valueHex).toString()
                
                transfers.push({
                    contract: tx.to,
                    from: fromHex,
                    to: toHex,
                    value: valueDex,
                    type: "transferFrom"
                })
            }
        }
        console.log(transfers);
    } catch (error) {
        console.error('Error parsing transaction:', error)
    }
}
