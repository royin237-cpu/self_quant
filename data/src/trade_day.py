import baostock as bs
import pandas as pd

# 缓存一份交易日历，避免多次调用API
_trade_dates_cache = None


def get_valid_trade_date(target_date, direction='forward'):
    """
    target_date: 目标日期字符串 'YYYY-MM-DD'
    direction: 'forward' 表示向后找（如果今天是节假日，找下一个交易日）
               'backward' 表示向前找（如果今天是节假日，找上一个交易日）
    """
    global _trade_dates_cache
    if _trade_dates_cache is None:
        # 获取从2010年到2030年的所有交易日
        rs = bs.query_trade_dates(start_date="2010-01-01", end_date="2030-12-31")
        print('在运行get_valid_trade_date')
        all_dates = rs.get_data()
        _trade_dates_cache = all_dates[all_dates['is_trading_day'] == '1']['calendar_date'].tolist()

    # 将交易日列表转为 Series 以便使用 searchsorted
    dates_series = pd.Series(_trade_dates_cache)

    if direction == 'forward':
        # 寻找第一个 >= target_date 的交易日
        idx = dates_series.searchsorted(target_date)
    else:
        # 寻找第一个 <= target_date 的交易日
        idx = dates_series.searchsorted(target_date, side='right') - 1

    if 0 <= idx < len(dates_series):
        return dates_series.iloc[idx]
    else:
        return target_date  # 如果超出范围，返回原日期

if __name__ == '__main__':
    bs.login()
    print(get_valid_trade_date('2026-03-21', 'backward'))
    bs.logout()
