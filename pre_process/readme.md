# pre_process

这一部分代码用于**对原始交易的 EVM call trace 做预处理**，把复杂的嵌套执行过程整理成**后续可建模的调用序列**。


## 目录结构

```text
pre_process/
├── index.js
└── trace/
    ├── linearizeTrace.js
    └── buildCallSequence.js
```


## 各文件作用

### `index.js`

预处理的**入口文件**。

* 组织整个 trace 预处理流程
* 调用 trace 相关函数
* 输出单笔交易对应的 call 序列结果


### `trace/linearizeTrace.js`

**把树状的 call trace 展平为线性序列**。

* 输入：嵌套的 EVM call trace（包含 internal calls）
* 输出：按执行顺序排列的一维 trace 节点列表
* 主要作用：

  * 消除嵌套结构
  * 保留调用的先后顺序和层级信息（depth）


### `trace/buildCallSequence.js`

**从 trace 节点构造标准化的 call event**。

* 输入：线性化后的 trace 节点
* 输出：call event 序列（JSON）
* 每个 call event 只保留与“调用行为”相关的信息，例如：

  * call 类型
  * 合约地址
  * 函数 selector / 输入摘要
  * depth、value、gas、是否 revert 等


## 输出结果

`pre_process` 的最终输出是：

* **一个按执行顺序排列的 call event 数组**
* 不涉及 token、vocab 或模型
* 作为 `representation/tokenization` 的直接输入
