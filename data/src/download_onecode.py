#%%
import baostock as bs
import pandas as pd

from functools import reduce # 用于合并多个表
from trade_day import get_valid_trade_date

def get_stock_data(code, start_date, end_date):
    # 起始日期找“之后”的第一个交易日，结束日期找“之前”的最后一个交易日
    start_date = get_valid_trade_date(start_date, direction='forward')
    end_date = get_valid_trade_date(end_date, direction='backward')
    print(f"校准后的日期范围: {start_date} 至 {end_date}")
    # 2. 获取K线数据
    rs_k = bs.query_history_k_data_plus(code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")
    df_k = rs_k.get_data()

    df_k['date'] = pd.to_datetime(df_k['date'])

    # --- 自动调整年份 ---
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years = [str(y) for y in range(start_year - 1, end_year + 1)] # 建议多取一年以确保覆盖
    quarters = ['1', '2', '3', '4']

    # 定义财务数据查询函数
    def get_financial_data(query_func, years, quarters):
        all_data = pd.DataFrame()
        for y in years:
            for q in quarters:
                rs = query_func(code=code, year=y, quarter=q)
                data = rs.get_data()
                if not data.empty:
                    all_data = pd.concat([all_data, data], ignore_index=True)
        return all_data

    # 3. 获取各维度财务数据
    # 注意：移除fields参数，因为baostock的query方法不需要传入字段字符串，它返回固定结构
    profit = get_financial_data(bs.query_profit_data, years, quarters)
    growth = get_financial_data(bs.query_growth_data, years, quarters)
    operation = get_financial_data(bs.query_operation_data, years, quarters)
    debt = get_financial_data(bs.query_balance_data, years, quarters)
    cash = get_financial_data(bs.query_cash_flow_data, years, quarters)
    dupont = get_financial_data(bs.query_dupont_data, years, quarters)

    # 4. 合并财务数据 (使用 reduce 批量合并)
    dfs = [profit, growth, operation, debt, cash, dupont]


    # 将列表中的表依次按照公共列进行左连接
    fin_df = reduce(lambda left, right: pd.merge(left, right, on=['code', 'pubDate', 'statDate'], how='outer'), dfs)

    fin_df['pubDate'] = pd.to_datetime(fin_df['pubDate'])
    fin_df = fin_df.sort_values('pubDate')

    # 5. 使用 merge_asof 将财务数据匹配到K线
    final_df = pd.merge_asof(df_k.sort_values('date'),
                             fin_df.sort_values('pubDate'),
                             left_on='date',
                             right_on='pubDate',
                             by='code',
                             direction='backward')

    return final_df


if __name__ == '__main__':
    bs.login()
    # --- 执行示例 ---
    code = "sz.002424"
    df_result = get_stock_data(code, "2024-01-01", "2026-3-21")
    # print(df_result.tail())
    #输出csv文件

    df_result.to_csv(f'{code}.csv', index=False)
    bs.logout()
