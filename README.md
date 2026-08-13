# Self Quant — A 股个人量化交易研究平台

从数据采集、策略回测到实盘下单的完整量化投研闭环，聚焦沪深主板，以小市值低价轮动为核心策略，逐步扩展至技术指标、基本面价值、动量、机器学习及学术因子研究等方向。

## 项目架构

```
self_quant/
├── data/               # 数据层：全市场行情与财务数据采集、清洗、合并
├── quant/              # 策略实验室：30+ 策略回测 Notebook + 交易明细 Excel
├── quant_gemini/       # 因子研究：WorldQuant 101 Alpha 与国泰君安 191 因子框架
├── choose/             # 实时选股：双数据源（baostock + akshare）实时筛选
├── order/              # 实盘交易：easytrader 对接同花顺自动化下单
├── 因子/               # 理论参考：WorldQuant 论文与国泰君安因子报告
└── README.md
```

## 核心工作流

```
数据采集  ──►  数据合并  ──►  策略回测  ──►  结果输出
baostock       K线+财报        30+ Notebook     交易明细 Excel
akshare        合并为大表       因子分组回测      收益曲线 PNG
               (1.6GB)         机器学习预测      前向收益 CSV
                                    │
                    实时选股  ──►  实盘交易
                    choose/       order/
                    双源筛选       easytrader+同花顺
```

## 目录详解

### data/ — 数据层

负责全市场 A 股数据的下载、清洗与合并，输出一张包含 K 线 + 三大财报的完整面板数据表。

| 关键文件 | 功能 |
|---|---|
| `src/download_onecode.py` | 通过 baostock 下载单只股票日 K 线 + 6 大财务报表，用 `merge_asof` 匹配财报到 K 线（防止未来函数） |
| `src/download_all.py` | 批量下载全市场股票，同时存入 CSV 和 MySQL |
| `src/stock_pool.py` | 获取 A 股股票池，支持按代码前缀过滤（沪市 60、深市 00 等） |
| `src/trade_day.py` | 交易日历工具，查找目标日期前后最近的交易日 |
| `notebook/merge_all_data.py` | **核心合并脚本**：日 K 线 + 利润表 + 资产负债表 + 现金流量表合并，处理退市整理期，计算流通股本和流通市值，输出 `all_stocks_merged_fixed.parquet`（约 1.6GB，45 字段） |

**合并大表字段（45 列）：**

```
date, code, open, high, low, close, preclose, volume, amount, adjustflag,
turn, tradestatus, pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM, isST,
报告日, 净利润, 营业利润, 利润总额, 应收账款, 存货, 资产总计, 负债合计,
经营活动产生的现金流量净额, 投资活动产生的现金流量净额, 筹资活动产生的现金流量净额,
现金及现金等价物净增加额, 实际披露日, stock_status,
营业总收入, 营业总成本, 营业成本, 所得税费用, 归属于母公司所有者的净利润,
货币资金, 流动资产合计, 非流动资产合计, 流动负债合计, 非流动负债合计,
所有者权益(或股东权益)合计, 流通股本, 流通市值
```

### quant/ — 策略回测实验室

包含 30+ 个策略的 Jupyter Notebook 回测脚本，是项目的核心策略开发目录。

**策略分类：**

| 类别 | 代表策略 | 说明 |
|---|---|---|
| **小市值轮动** | `小市值+每周调仓.ipynb` | 从流通市值最小的 50 只中选价格最低的 10 只，每周调仓 |
| | `小市值_周调仓_清理退市_v2.ipynb` | 增加退市清理逻辑的优化版 |
| | `小市值+低价+pe大于0.ipynb` | 增加 PE > 0 过滤 |
| | `小市值_低估值_归母净利同比增长大于百分之10.ipynb` | 增加利润增长过滤 |
| **技术指标** | `MACD_RSI_近零轴优先策略.ipynb` | MACD 近零轴金叉买入 + RSI 超买卖出 |
| | `MACD主升浪捕捉+倍投网格+胜率自适应.ipynb` | MACD + 马丁格尔倍投 + 胜率自适应 |
| | `筹码峰.ipynb` | 换手率衰减筹码分布计算 + 筹码峰突破选股 |
| | `放量突破.ipynb` | 60 日新高 + 2 倍放量突破 |
| | `横盘择时.ipynb` | 线性回归斜率判定趋势通道 |
| **基本面/价值** | `巴菲特策略.ipynb` | ROE > 15%、盈利含金量 > 0.8、半年调仓 |
| | `高ROE长线白马.ipynb` | 高 ROE + 低 PE + 现金流正 |
| | `红利低波.ipynb` | 低 PE + 低 60 日波动率 |
| **动量** | `20日动量_后10名.ipynb` | 强势股中滞涨股策略 |
| **机器学习** | `机器学习.ipynb` | 随机森林分类器，7 特征预测未来 20 天涨跌 |
| | `机器学习_xgboost.ipynb` | XGBoost（GPU 加速）批量预测选股 |

### quant_gemini/ — 因子研究模块

使用专业因子评价框架进行因子有效性检验，包含五分组回测和 Rank IC 分析。

| 文件 | 功能 |
|---|---|
| `code/worldquant_alphas_test.ipynb` | WorldQuant 101 Alpha 多因子评价框架，计算 Rank IC、IC IR、五分组收益差，评估 9 个因子 |
| `code/worldquant_alpha003_backtest.ipynb` | Alpha 003 因子五分组回测 |
| `code/gtja_191_alphas_backtest.ipynb` | 国泰君安 191 因子集回测 |
| `code/momentum_60d_backtest.ipynb` | 60 日动量因子五分组回测 |
| `code/growth_low_price_strategy.py` | 高成长 + 低价策略 |
| `code/small_cap_weekly_with_metrics.py` | 小市值周轮动 + 夏普比率、最大回撤等风险指标 |
| `GEMINI.md` | 因子研究回测规范文档 |

### choose/ — 实时选股

通过双数据源获取全市场实时行情，筛选市值最小 + 价格最低的股票。

- `choose.ipynb`：baostock 数据源
- `akshare_choose.ipynb`：akshare 东方财富数据源，含代理清理和重试机制

### order/ — 实盘交易

通过 `easytrader` 库连接同花顺客户端，实现自动化买入/卖出/撤单/查询持仓等功能。

### 因子/ — 理论参考

| 文件 | 说明 |
|---|---|
| `101 Formulaic Alphas.pdf` | WorldQuant 101 Formulaic Alphas 论文 |
| `国泰君安－基于短周期价量特征的多因子选股体系.pdf` | 国泰君安 191 因子研究报告 |

## 技术栈

| 库 | 用途 |
|---|---|
| **pandas / numpy** | 数据处理与数值计算 |
| **baostock** | A 股历史 K 线和财务数据下载（主要数据源） |
| **akshare** | A 股实时行情和财务报表下载（辅助数据源） |
| **pyarrow** | Parquet 文件读写 |
| **matplotlib** | 收益曲线可视化（SimHei 中文支持） |
| **scikit-learn** | 随机森林分类器 |
| **xgboost** | XGBoost 分类器（GPU 加速） |
| **easytrader** | 同花顺自动化下单 |
| **sqlalchemy + pymysql** | MySQL 数据库存储 |
| **tqdm** | 批量下载进度条 |

## 数据概况

- **数据来源**：baostock（历史数据）、akshare（实时数据 + 退市名单）
- **覆盖范围**：全市场 A 股约 5000+ 只股票（含退市票）
- **时间跨度**：2010-01-01 至 2026-03-20
- **核心数据文件**：`all_stocks_merged_fixed.parquet`（约 1.6GB，45 字段，数百万行面板数据）
- **数据类型**：日 K 线、利润表、资产负债表、现金流量表、大盘指数、退市票数据

## 回测规范

| 参数 | 设定 |
|---|---|
| 初始资金 | 50,000 元 |
| 目标持仓 | 5–10 只 |
| 调仓频率 | 每周 / 每月 / 半年（因策略而异） |
| 佣金 | 0.025%（最低 5 元） |
| 过户费 | 0.001% |
| 印花税 | 0.05%（仅卖出） |
| 涨停限制 | > 9.3% 无法买入 |
| 跌停限制 | < -9.3% 无法卖出 |
| 退市处理 | 计提 100% 亏损 |
| 无风险利率 | 3%（夏普比率计算用） |

## 快速开始

### 1. 环境准备

```bash
pip install pandas numpy matplotlib baostock akshare pyarrow tqdm \
             scikit-learn xgboost sqlalchemy pymysql easytrader
```

### 2. 数据下载与合并

```bash
# 下载全市场数据（K线 + 财报）
python data/src/download_all.py

# 合并为大表
python data/notebook/merge_all_data.py
# 输出：data/data/all_stocks_merged_fixed.parquet
```

### 3. 策略回测

在 Jupyter Notebook 中打开 `quant/` 目录下的任意策略文件，加载合并大表即可运行回测：

```python
import pandas as pd
df = pd.read_parquet('data/data/all_stocks_merged_fixed.parquet')
```

### 4. 实时选股

运行 `choose/` 目录下的 Notebook，获取当日选股结果。

### 5. 实盘交易

配置同花顺客户端后，运行 `order/trade.ipynb` 进行自动化交易。

## 项目结构总览

```
self_quant/
├── data/
│   ├── src/                    # 数据下载脚本 (Python)
│   ├── notebook/               # 数据处理 Notebook
│   ├── data/                   # 完整数据存储 (gitignore)
│   │   ├── all_stocks_merged_fixed.parquet   # 核心合并大表
│   │   ├── 日K线/  利润/  资产负债/  现金流量/
│   │   └── 退市票*/  合并后数据*/  指数/
│   └── samples/                # 样本数据（每类 5 个，纳入 Git）
├── quant/
│   ├── *.ipynb                 # 30+ 策略回测 Notebook
│   ├── *.py                    # 策略脚本
│   ├── 回测excel/              # 交易明细 Excel
│   └── strategy_summary.txt    # 策略摘要
├── quant_gemini/
│   ├── code/                   # 因子回测 Notebook 和脚本
│   ├── csv/                    # 因子回测结果
│   ├── cache/                  # 缓存
│   └── GEMINI.md               # 回测规范文档
├── choose/                     # 实时选股 Notebook
├── order/                      # 实盘交易 Notebook
├── 因子/                       # 因子理论 PDF
└── .gitignore
```

## 风险提示

本项目仅供个人学习和研究使用，不构成任何投资建议。量化交易存在实质性风险，历史回测表现不代表未来收益。实盘交易模块请谨慎使用，充分了解相关风险后方可启用。
