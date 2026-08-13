# 多因子选股策略设计（两套独立方案）

## 项目概述
本项目在 A 股 2010 年至今的全市场数据（`F:\self_quant\data\data\all_stocks_merged_fixed.parquet`）上实现 **两套独立的短周期价量特征多因子选股策略**，分别对应方案 2（LightGBM 梯度提升回归）和方案 3（熵权‑排名聚合）。每套方案各自完成因子计算、自动权重生成、周调仓、费用及涨跌停处理，并输出独立的回测报告、净值曲线和交易明细，便于后续对比分析。

---

## 1. 共同基础设施
- **数据来源**：Parquet 文件，列定义见 `cols.txt`。
- **因子子集**：从 `因子/国泰君安－基于短周期价量特征的多因子选股体系.pdf` 中取前 ⅓（约 63）个因子，列于 `factors/factor_definitions.txt`。
- **交易规则**
  - 初始资金 50,000 RMB。
  - 持仓上限 10 只股票。
  - 每周调仓（当周最后交易日）。
  - 费用模型：佣金 0.025%（最低 5 元），过户费 0.001%，卖出印花税 0.05%。
  - 跌停 (<‑9.3%) 时不可卖出，涨停 (>9.3%) 时不可买入。
  - 数据缺失视为退市，持仓在退市日计提 100% 亏损。
- **输出**（每套方案分别在 `output/lightgbm/` 与 `output/entropy/`）
  - `trades.csv`：买卖点明细（含因子得分、前向收益等）。
  - `equity_curve.png`：净值曲线。
  - `report.html`：交互式报告，包含累计收益、年化收益、最大回撤、夏普比、持仓换手率以及因子权重/熵值可视化。

---

## 2. 方案 2 – LightGBM 梯度提升回归
### 2.1 关键模块
- `src/data_loader.py` – 读取 parquet、时间列转 datetime、退市标记。
- `src/factor_engine.py` – 计算 63 个短周期价量因子并进行 Z‑score 标准化。
- `impl_lightgbm/model_trainer.py`
  - 使用最近 120 天（约 6 个月）滚动窗口训练 **3 个 LGBMRegressor**（目标分别为持仓 1 天、1 周、1 月的前向收益率）。
  - 参数示例：`learning_rate=0.05`, `num_leaves=31`, `metric='rmse'`，采用交叉验证防止过拟合。
- `impl_lightgbm/scorer_lgb.py`
  - 对每只股票进行预测，取三条预测收益的等权平均得到 `lgb_score`。
  - 使用模型训练结束后的 `feature_importances_`（基于 Gain）归一化得到每个因子的自动权重，可在报告中展示。
- `src/portfolio.py` & `src/backtest.py`
  - 选股依据 `lgb_score`（从高到低取前 10 只），其余业务逻辑与统一规则相同。

### 2.2 结果输出路径
- `output/lightgbm/trades.csv`
- `output/lightgbm/equity_curve.png`
- `output/lightgbm/report.html`

---

## 3. 方案 3 – 熵权‑排名聚合
### 3.1 关键模块
- `src/data_loader.py`、`src/factor_engine.py` 同上。
- `impl_entropy/scorer_entropy.py`
  - 对每个因子进行 **升/降序排名**（依据因子含义），得到归一化排名 `r_ij`。
  - 计算信息熵 `e_j = -k Σ_i p_ij ln(p_ij)`，其中 `p_ij = r_ij / Σ_i r_ij`，`k` 为归一化常数。
  - 熵权 `w_j = (1‑e_j) / Σ (1‑e_j)`，随因子区分度自适应。
  - 综合得分 `entropy_score = Σ_j w_j * r_ij`，得分越高表示因子综合实力越强。
- `src/portfolio.py` & `src/backtest.py`
  - 选股依据 `entropy_score`（从高到低取前 10），其余业务逻辑保持一致。

### 3.2 结果输出路径
- `output/entropy/trades.csv`
- `output/entropy/equity_curve.png`
- `output/entropy/report.html`

---

## 4. 回测统计指标（统一）
| 指标 | 计算方式 |
|------|----------|
| **累计收益率** | `(最终资产‑初始资产) / 初始资产 × 100%` |
| **年化收益率** | 基于每周复利换算 |
| **最大回撤** | 资产净值相对历史最高点的最大跌幅 |
| **夏普比** | `(年化收益‑无风险利率) / 年化波动率`（默认无风险率 0） |
| **持仓周换手率** | `(本周卖出+买入股票数) / 当周持仓数` |
| **因子贡献** | LightGBM：特征重要性占比；熵权：各因子熵权占比 |
| **买卖点 CSV** | 包含 `日期、代码、方向、成交价、费用、因子得分、前向收益(1d/1w/1m)` |

---

## 5. 任务清单（TodoWrite）
- [ ] 实现 `data_loader.py`（读取 parquet、退市标记）。
- [ ] 编写 `factor_engine.py`（63 因子计算、标准化）。
- **方案 2**
  - [ ] `impl_lightgbm/model_trainer.py`（滚动窗口 LightGBM 训练）。
  - [ ] `impl_lightgbm/scorer_lgb.py`（预测与得分）。
- **方案 3**
  - [ ] `impl_entropy/scorer_entropy.py`（熵权计算）。
- [ ] `portfolio.py`（持仓管理、费用、涨跌停、退市亏损）。
- [ ] `backtest.py`（回测循环、结果统计、报告生成）。
- [ ] 生成 `output/*` 中的 CSV、PNG、HTML 报告。
- [ ] 将本设计文档提交至 `docs/superpowers/specs/2026-05-07-multi-factor-selection-design.md` 并记录在 `MEMORY.md`（后续可手动完成）。

---

**下一步**：我将使用 `writing-plans` 技能为 **方案 2** 和 **方案 3** 分别生成详细的实现计划。请确认以上设计是否可以开始。