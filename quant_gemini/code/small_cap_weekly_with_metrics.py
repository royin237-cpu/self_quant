import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 设置绘图支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 第一步：配置参数与读取数据
# ==========================================
FILE_PATH = r"F:\self_quant\data\data\all_stocks_merged_fixed.parquet"
OUTPUT_EXCEL = r"F:\self_quant\quant\回测excel\小市值低价轮动策略明细_每周_含风险指标.xlsx"
OUTPUT_IMAGE = r"F:\self_quant\quant_gemini\csv\small_cap_weekly_with_metrics.png"

INITIAL_CASH = 50000.0  # 初始本金 5万元
TARGET_STOCK_NUM = 10   # 目标持仓 10 只
RISK_FREE_RATE = 0.03   # 无风险利率 3%

def calculate_fees(trade_type, price, volume):
    amount = price * volume
    commission = max(5.0, amount * 0.00025)
    transfer_fee = amount * 0.00001
    stamp_duty = amount * 0.0005 if trade_type == 'sell' else 0.0
    total_fee = commission + transfer_fee + stamp_duty
    return total_fee, amount

class Account:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.positions = {}
        self.trade_logs = []
        self.history_assets = []

    def get_total_asset(self, current_date, current_prices_dict):
        stock_value = 0
        for code, pos in self.positions.items():
            # 如果当天没有价格，使用持仓中的最后记录价
            price = current_prices_dict.get(code, pos['price'])
            self.positions[code]['price'] = price
            stock_value += pos['volume'] * price
        return self.cash + stock_value

def run_backtest():
    print("正在读取全市场数据，请稍候...")
    if not os.path.exists(FILE_PATH):
        print(f"错误：找不到数据文件 {FILE_PATH}")
        return

    df = pd.read_parquet(FILE_PATH)
    df['date'] = pd.to_datetime(df['date'])

    # 过滤沪深主板股票
    print("正在过滤沪深主板数据...")
    main_board_mask = df['code'].str.contains(r'^sh\.60|^sz\.00', regex=True)
    df = df[main_board_mask].copy()

    df.sort_values(by=['date', 'code'], inplace=True)

    # 提取每周的最后一个交易日作为调仓日
    df['year_week'] = df['date'].dt.to_period('W')
    trade_dates = df.groupby('year_week')['date'].max().sort_values().tolist()

    account = Account(INITIAL_CASH)

    print("开始按周执行小市值+低价股调仓模拟...")

    for date in trade_dates:
        daily_data = df[df['date'] == date].set_index('code')
        current_prices = daily_data['close'].to_dict()
        status_dict = daily_data['tradestatus'].to_dict()
        st_dict = daily_data['isST'].to_dict()
        pct_chg_dict = daily_data['pctChg'].to_dict()

        current_asset = account.get_total_asset(date, current_prices)
        account.history_assets.append({'date': date, 'total_asset': current_asset})

        # 选股逻辑
        valid_pool = daily_data[(daily_data['tradestatus'] == 1) &
                                (daily_data['isST'] == 0) &
                                (daily_data['peTTM'] > 0) & (daily_data['pbMRQ'] > 0)].copy()

        target_codes = []
        if len(valid_pool) >= 50:
            smallest_50 = valid_pool.sort_values(by='流通市值', ascending=True).head(50).copy()
            target_codes = smallest_50.sort_values(by='close', ascending=True).head(TARGET_STOCK_NUM).index.tolist()

        # ---------------- 卖出逻辑 ----------------
        current_holdings = list(account.positions.keys())
        for code in current_holdings:
            if code not in daily_data.index:
                # 处理退市
                lost_vol = account.positions[code]['volume']
                last_price = account.positions[code]['price']
                loss_amount = lost_vol * last_price
                del account.positions[code]
                account.trade_logs.append({
                    '日期': date, '股票代码': code, '买卖方向': '强制清理(退市)',
                    '成交价': 0, '成交股数': lost_vol, '成交金额': 0, '交易费': 0,
                    '可用现金': round(account.cash, 2),
                    '总资产': round(account.get_total_asset(date, current_prices), 2),
                    '备注': f'数据缺失/退市，计提损失 {round(loss_amount, 2)}元'
                })
                continue
            
            if code not in target_codes:
                is_trading = status_dict.get(code, 0) == 1
                pct_chg = pct_chg_dict.get(code, 0)

                if is_trading and pct_chg > -9.3:
                    sell_price = current_prices[code]
                    sell_vol = account.positions[code]['volume']
                    fee, amount = calculate_fees('sell', sell_price, sell_vol)

                    account.cash += (amount - fee)
                    del account.positions[code]

                    account.trade_logs.append({
                        '日期': date, '股票代码': code, '买卖方向': '卖出',
                        '成交价': sell_price, '成交股数': sell_vol,
                        '成交金额': round(amount, 2), '交易费': round(fee, 2),
                        '可用现金': round(account.cash, 2),
                        '总资产': round(account.get_total_asset(date, current_prices), 2),
                        '备注': '正常卖出'
                    })

        # ---------------- 买入逻辑 ----------------
        stocks_to_buy = [code for code in target_codes if code not in account.positions]
        free_slots = max(0, TARGET_STOCK_NUM - len(account.positions))
        actual_stocks_to_buy = stocks_to_buy[:free_slots]

        if len(actual_stocks_to_buy) > 0:
            target_cash_per_stock = account.cash / len(actual_stocks_to_buy)
            for code in actual_stocks_to_buy:
                is_trading = status_dict.get(code, 0) == 1
                pct_chg = pct_chg_dict.get(code, 0)
                is_st = st_dict.get(code, 0) == 1

                if is_trading and not is_st and pct_chg < 9.3:
                    buy_price = current_prices[code]
                    affordable_shares = int((target_cash_per_stock * 0.997) / buy_price)
                    buy_vol = (affordable_shares // 100) * 100

                    if buy_vol >= 100:
                        fee, amount = calculate_fees('buy', buy_price, buy_vol)
                        if account.cash >= (amount + fee):
                            account.cash -= (amount + fee)
                            account.positions[code] = {'volume': buy_vol, 'price': buy_price}
                            account.trade_logs.append({
                                '日期': date, '股票代码': code, '买卖方向': '买入',
                                '成交价': buy_price, '成交股数': buy_vol,
                                '成交金额': round(amount, 2), '交易费': round(fee, 2),
                                '可用现金': round(account.cash, 2),
                                '总资产': round(account.get_total_asset(date, current_prices), 2),
                                '备注': '正常买入'
                            })

    # ==========================================
    # 第四步：计算风险指标
    # ==========================================
    df_assets = pd.DataFrame(account.history_assets)
    df_assets.set_index('date', inplace=True)
    
    # 计算收益率
    df_assets['returns'] = df_assets['total_asset'].pct_change().fillna(0)
    
    # 1. 总收益率
    final_asset = df_assets['total_asset'].iloc[-1]
    total_return = (final_asset - INITIAL_CASH) / INITIAL_CASH
    
    # 2. 年化收益率 (假设一年52周)
    num_weeks = len(df_assets)
    annual_return = (1 + total_return) ** (52 / num_weeks) - 1
    
    # 3. 夏普比率 (周度转年化)
    # 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
    weekly_std = df_assets['returns'].std()
    annual_volatility = weekly_std * np.sqrt(52)
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility != 0 else 0
    
    # 4. 最大回撤
    df_assets['cum_max'] = df_assets['total_asset'].cummax()
    df_assets['drawdown'] = (df_assets['total_asset'] - df_assets['cum_max']) / df_assets['cum_max']
    max_drawdown = df_assets['drawdown'].min()
    
    # 5. 最大回撤持续时间
    # 找到回撤点
    is_in_drawdown = df_assets['drawdown'] < 0
    # 计算连续回撤的长度
    drawdown_periods = []
    current_period = 0
    for in_drawdown in is_in_drawdown:
        if in_drawdown:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0
    if current_period > 0:
        drawdown_periods.append(current_period)
    
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    
    # 打印结果
    print("\n" + "="*30)
    print("      策略回测风险指标")
    print("="*30)
    print(f"初始本金: {INITIAL_CASH:,.2f}")
    print(f"最终资产: {final_asset:,.2f}")
    print(f"总收益率: {total_return*100:.2f}%")
    print(f"年化收益率: {annual_return*100:.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"最大回撤持续周数: {max_drawdown_duration} 周")
    print(f"年化波动率: {annual_volatility*100:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print("="*30)

    # 保存结果
    df_trades = pd.DataFrame(account.trade_logs)
    df_trades.to_excel(OUTPUT_EXCEL, index=False)
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 资产净值
    ax1 = plt.subplot(211)
    ax1.plot(df_assets.index, df_assets['total_asset'], label='策略净值', color='blue')
    ax1.set_title(f"小市值周轮动策略回测 (总收益: {total_return*100:.2f}%)")
    ax1.set_ylabel("总资产")
    ax1.legend()
    ax1.grid(True)

    # 子图2: 回撤
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.fill_between(df_assets.index, 0, df_assets['drawdown'], color='red', alpha=0.3, label='回撤')
    ax2.set_title(f"策略回撤图 (最大回撤: {max_drawdown*100:.2f}%)")
    ax2.set_ylabel("回撤幅度")
    ax2.set_xlabel("日期")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"✅ 收益曲线图已保存至: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    run_backtest()
