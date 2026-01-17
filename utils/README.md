## Track Tokens
- LINK: `0x514910771AF9Ca656af840dff83E8264EcF986CA`
- COMP: `0xc00e94Cb662C3520282E6f5717214004A7f26888`
- UNI:  `0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984`
- CRV:  `0xD533a949740bb3306d119CC777fa900bA034cd52`, Curve DAO Token
- 3CRV: `0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490`, 3pool LP Token
- MKR:  `0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2`
- WBTC: `0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599`, Wrapped Bitcoin

## 本地跑获取外部交易记录
```
node .\utils\getTokenTransactions.js CONTRACT_ADDRESS
```

示例
```
node .\utils\getTokenTransactions.js 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599
```
## PM2命令(Node.js 长期运行)
```
pm2 start path/to/script.js \
  --name my-task \
  -- arg1 arg2
```
示例：
```
pm2 start utils/getTokenTransactions.js \
  --name wbtc-scanner \
  -- 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599
```
开机自启（必须做）
```
pm2 startup
pm2 save
```
查看日志
```
pm2 logs my-task
```
常用控制命令
```
pm2 stop my-task
pm2 restart my-task
pm2 delete my-task
```