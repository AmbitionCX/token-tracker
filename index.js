import { provider } from './utils/provider.js';
import dotenv from 'dotenv';
dotenv.config();

provider.getBlockNumber().then( res => {
    console.log(res);
})