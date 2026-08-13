import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置绘图支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FILE_PATH = r"F:\self_quant\data\data\all_stocks_merged_fixed.parquet"
OUTPUT_CSV = r"F:\self_quant\quant_gemini\csv\growth_low_price_trades.csv"

INITIAL_CASH = 50000.0  # 初始本金
TARGET_STOCK_NUM = 5    # 买入价格最低的5只股票

print("正在读取全市场数据...")
df = pd.read_parquet(FILE_PATH)
df['date'] = pd.to_datetime(df['date'])
df['实际披露日'] = pd.to_datetime(df['实际披露日'])

# 过滤沪深主板股票
main_board_mask = df['code'].str.contains(r'^sh\.60|^sz\.00', regex=True)
df = df[main_board_mask].copy()
df.sort_values(by=['code', 'date'], inplace=True)

print(f"数据范围: {df['date'].min().date()} 至 {df['date'].max().date()}")

# ==========================================
# 第一步：计算前向收益
# ==========================================
print("正在计算前向收益率...")
if 'code' not in df.columns:
    df.reset_index(inplace=True)
df['fwd_ret_1d'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
df['fwd_ret_1w'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
df['fwd_ret_1m'] = df.groupby('code')['close'].shift(-20) / df['close'] - 1

# ==========================================
# 第二步：计算财务成长指标
# ==========================================
print("正在计算基本面增长率(YoY和CAGR)...")
# 提取非空财报披露点
fund_df = df[df['实际披露日'].notna()].drop_duplicates(subset=['code', '实际披露日']).copy()
fund_df.sort_values(by=['code', '实际披露日'], inplace=True)

# 1. 营业收入同比增长率 (营业总收入 YoY)
fund_df['营收YoY'] = fund_df.groupby('code')['营业总收入'].pct_change(periods=4)

# 2. 总利润同比增长率 (利润总额 YoY)
fund_df['利润总额YoY'] = fund_df.groupby('code')['利润总额'].pct_change(periods=4)

# 3. 营业利润复合年均增长率（3年）
# 假设财报每季度公布，3年对应12个季度(periods=12)
fund_df['营业利润_3年前'] = fund_df.groupby('code')['营业利润'].shift(12)
# CAGR计算需要基数为正数，否则数学意义失效。我们将基数为负的情况剔除(设为NaN)
fund_df['营业利润CAGR3'] = np.where(
    fund_df['营业利润_3年前'] > 0,
    (fund_df['营业利润'] / fund_df['营业利润_3年前']) ** (1/3) - 1,
    np.nan
)

# 将计算好的指标合并回大表
merge_cols = ['code', '实际披露日', '营收YoY', '利润总额YoY', '营业利润CAGR3']
df = pd.merge(df, fund_df[merge_cols], on=['code', '实际披露日'], how='left')

# NaN处理：财务数据按时间向下填充(ffill)
df.sort_values(by=['code', 'date'], inplace=True)
df['营收YoY'] = df.groupby('code')['营收YoY'].ffill()
df['利润总额YoY'] = df.groupby('code')['利润总额YoY'].ffill()
df['营业利润CAGR3'] = df.groupby('code')['营业利润CAGR3'].ffill()

# 提取每周最后一个交易日作为调仓日
df['year_week'] = df['date'].dt.to_period('W')
trade_dates = df.groupby('year_week')['date'].max().unique()
trade_dates = sorted(trade_dates.tolist())

# ==========================================
# 第三步：交易费用及账户类
# ==========================================
def calculate_fees(trade_type, price, volume):
    amount = price * volume
    commission = max(5.0, amount * 0.00025)
    transfer_fee = amount * 0.00001
    stamp_duty = amount * 0.0005 if trade_type == 'sell' else 0.0
    return commission + transfer_fee + stamp_duty, amount

class Account:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.positions = {}
        self.trade_logs = []
        self.history_assets = []

    def get_total_asset(self, current_prices_dict):
        stock_value = 0
        for code, pos in self.positions.items():
            price = current_prices_dict.get(code, pos['price'])
            stock_value += pos['volume'] * price
        return self.cash + stock_value

account = Account(INITIAL_CASH)

# ==========================================
# 第四步：执行回测
# ==========================================
print("开始执行每周轮动回测...")
for date in trade_dates:
    daily_data = df[df['date'] == date].set_index('code')
    if daily_data.empty: continue
    
    current_prices = daily_data['close'].to_dict()
    status_dict = daily_data['tradestatus'].to_dict()
    st_dict = daily_data['isST'].to_dict()
    pct_chg_dict = daily_data['pctChg'].to_dict()
    
    # 记录净值
    current_asset = account.get_total_asset(current_prices)
    account.history_assets.append({'date': date, 'total_asset': current_asset})
    
    # 选股：
    # 1. 有效交易且非ST
    # 2. 营业收入增长率 >= 20%
    # 3. 总利润同比增长率 >= 40%
    # 4. 营业利润复合年均增长率(3年) > 20%
    valid_mask = (
        (daily_data['tradestatus'] == 1) & 
        (daily_data['isST'] == 0) &
        (daily_data['营收YoY'] >= 0.20) &
        (daily_data['利润总额YoY'] >= 0.40) &
        (daily_data['营业利润CAGR3'] > 0.20)
    )
    valid_pool = daily_data[valid_mask].copy()
    
    # 选取买入价格最低的5只股票
    target_codes = valid_pool.sort_values(by='close', ascending=True).head(TARGET_STOCK_NUM).index.tolist()

    # ---------------- 卖出逻辑 ----------------
    current_holdings = list(account.positions.keys())
    for code in current_holdings:
        if code not in daily_data.index: # 退市处理
            lost_vol = account.positions[code]['volume']
            last_price = account.positions[code]['price']
            del account.positions[code]
            account.trade_logs.append({
                '日期': date, '代码': code, '方向': '退市清理', '成交价': 0, '成交额': 0, '费用': 0,
                '备注': f'计提亏损 {lost_vol*last_price:.2f}'
            })
            continue
            
        if code not in target_codes:
            is_trading = status_dict.get(code, 0) == 1
            pct_chg = pct_chg_dict.get(code, 0)
            
            # 跌停(<-9.3%)无法卖出
            if is_trading and pct_chg > -9.3:
                sell_price = current_prices[code]
                vol = account.positions[code]['volume']
                fee, amount = calculate_fees('sell', sell_price, vol)
                account.cash += (amount - fee)
                del account.positions[code]
                account.trade_logs.append({
                    '日期': date, '代码': code, '方向': '卖出', '成交价': sell_price, '成交额': amount, '费用': fee,
                    '备注': '调仓卖出'
                })

    # ---------------- 买入逻辑 ----------------
    stocks_to_buy = [c for c in target_codes if c not in account.positions]
    free_slots = TARGET_STOCK_NUM - len(account.positions)
    actual_buy_list = stocks_to_buy[:free_slots]
    
    if actual_buy_list:
        cash_per_stock = account.cash / len(actual_buy_list)
        for code in actual_buy_list:
            is_trading = status_dict.get(code, 0) == 1
            pct_chg = pct_chg_dict.get(code, 0)
            
            # 涨停(>9.3%)无法买入
            if is_trading and pct_chg < 9.3:
                buy_price = current_prices[code]
                vol = int((cash_per_stock * 0.997) / buy_price // 100) * 100
                if vol >= 100:
                    fee, amount = calculate_fees('buy', buy_price, vol)
                    if account.cash >= (amount + fee):
                        account.cash -= (amount + fee)
                        account.positions[code] = {'volume': vol, 'price': buy_price}
                        
                        # 记录买入及前向收益
                        row = daily_data.loc[code]
                        account.trade_logs.append({
                            '日期': date, '代码': code, '方向': '买入', '成交价': buy_price, '成交额': amount, '费用': fee,
                            '因子值': buy_price,  # 因子值为买入价格（越低越前）
                            '未来1天收益': row['fwd_ret_1d'],
                            '未来1周收益': row['fwd_ret_1w'],
                            '未来1月收益': row['fwd_ret_1m'],
                            '备注': '调仓买入'
                        })

print("回测完成。")

# ==========================================
# 第五步：结果分析与输出
# ==========================================
df_assets = pd.DataFrame(account.history_assets).set_index('date')
if df_assets.empty:
    print("未产生任何净值数据！")
else:
    final_asset = df_assets['total_asset'].iloc[-1]
    total_return = (final_asset - INITIAL_CASH) / INITIAL_CASH * 100

    buy_logs = pd.DataFrame(account.trade_logs)
    if not buy_logs.empty and '方向' in buy_logs.columns:
        buy_logs = buy_logs[buy_logs['方向'] == '买入']

    if not buy_logs.empty:
        stats = buy_logs[['未来1天收益', '未来1周收益', '未来1月收益']].mean() * 100
    else:
        stats = pd.Series({'未来1天收益': 0, '未来1周收益': 0, '未来1月收益': 0})

    print("\n--- 回测摘要 ---")
    print(f"初始本金: {INITIAL_CASH:,.2f}")
    print(f"最终资产: {final_asset:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"未来1天收益均值: {stats.get('未来1天收益', 0):.2f}%")
    print(f"未来1周收益均值: {stats.get('未来1周收益', 0):.2f}%")
    print(f"未来1月收益均值: {stats.get('未来1月收益', 0):.2f}%")

    if account.trade_logs:
        trades_df = pd.DataFrame(account.trade_logs)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        trades_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"交易明细已保存至: {OUTPUT_CSV}")

    # 绘制资金曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df_assets.index, df_assets['total_asset'], label=f"Total Asset (Return: {total_return:.2f}%)")
    plt.axhline(y=INITIAL_CASH, color='black', linestyle='--', alpha=0.5, label='初始资金')
    plt.title("高成长+低价轮动策略(营收 YoY>=20%, 利润 YoY>=40%, 利润3年CAGR>20%)")
    plt.xlabel("日期")
    plt.ylabel("总资产")
    plt.legend()
    plt.grid(True)
    
    img_path = r"F:\self_quant\quant_gemini\csv\growth_low_price_curve.png"
    plt.savefig(img_path)
    print(f"收益曲线已保存至: {img_path}")
