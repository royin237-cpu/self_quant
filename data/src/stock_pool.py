import baostock as bs
import datetime


def get_stock_pool(prefixes=['sh.60'], date=None):
    """
    获取A股股票池，可自定义股票代码前缀进行过滤

    Parameters:
    -----------
    prefixes : list, 可选
        需要保留的股票代码前缀列表。
        默认值为 ['sh.60', 'sz.00']，即沪市主板和深市主板。
    date : str/tuple/datetime, 可选
        查询日期，可以是字符串格式 'YYYY-MM-DD'，
        也可以是元组格式 (year, month, day)，
        或者 datetime 对象。如果为None，则使用今天。
    max_attempts : int, 可选
        向前查找的最大天数，默认30天。

    Returns:
    --------
    list
        符合条件的股票代码列表
    """


    max_attempts = 30

    # 处理 date 参数
    if date is None:
        date_obj = datetime.datetime.now()
    elif isinstance(date, str):
        try:
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            print(f"错误：日期格式不正确，应为 'YYYY-MM-DD'，当前为: {date}")
            return []
    elif isinstance(date, tuple) and len(date) == 3:
        # 如果是元组 (年, 月, 日)
        try:
            date_obj = datetime.datetime(date[0], date[1], date[2])
        except ValueError as e:
            print(f"错误：日期元组无效 {date}: {e}")
            return []
    elif isinstance(date, datetime.datetime):
        date_obj = date
    elif isinstance(date, datetime.date):
        date_obj = datetime.datetime.combine(date, datetime.time())
    else:
        print(f"错误：date 参数类型不支持: {type(date)}")
        return []

    # 将日期转换为字符串格式
    date_str = date_obj.strftime('%Y-%m-%d')
    print(f"查询日期: {date_str}")

    # 尝试获取数据，如果当天没有数据，则向前寻找
    for i in range(max_attempts):
        try:
            # 计算目标日期
            target_date = (date_obj - datetime.timedelta(days=i)).strftime('%Y-%m-%d')

            rs = bs.query_all_stock(day=target_date)
            print('在运行get_stock_pool查询所有股票')
            all_stocks = rs.get_data()

            # 检查是否有数据
            if all_stocks is not None and not all_stocks.empty:
                # 检查是否有正确的列（可能是'code'或'证券代码'等）
                if 'code' in all_stocks.columns:
                    code_column = 'code'
                elif '证券代码' in all_stocks.columns:
                    code_column = '证券代码'
                elif 'stock_code' in all_stocks.columns:
                    code_column = 'stock_code'
                elif len(all_stocks.columns) > 0:
                    # 尝试使用第一列
                    code_column = all_stocks.columns[0]
                else:
                    continue

                print(f" 截至{target_date} A股历史共有股票数据 {len(all_stocks)} 只股票")

                # 如果没有指定前缀或列表为空，返回所有股票
                if not prefixes:
                    return all_stocks[code_column].tolist()

                # 根据前缀列表构建过滤条件
                filter_condition = False
                for prefix in prefixes:
                    filter_condition = filter_condition | all_stocks[code_column].astype(str).str.startswith(prefix)

                # 应用过滤条件
                filtered_stocks = all_stocks[filter_condition]
                print(f"成功过滤后共获取 {len(filtered_stocks)} 只股票")
                return filtered_stocks[code_column].tolist()

        except Exception as e:
            print(f"查询 {target_date} 的数据时出错: {e}")
            continue

    # 如果经过所有尝试都没有找到数据
    print(f"在 {date_str} 及前 {max_attempts} 天内未找到有效股票数据")
    return []


# 使用示例
if __name__ == "__main__":
    bs.login()
    date = "2020-03-20"
    prefixes = ['sh.60', 'sz.00']
    # 1. 使用默认参数获取主板股票，未指定date是至今所有股票
    main_and_gem = get_stock_pool(prefixes=prefixes, date=date)
    bs.logout()

    # main_board_stocks = get_stock_pool()
    # # 或
    # main_board_stocks = get_stock_pool(prefixes=['sh.60', 'sz.00'])
    #
    # # 2. 只获取创业板股票
    # gem_stocks = get_stock_pool(prefixes=['sz.30'])
    #
    # # 3. 获取主板+创业板
    #
    # # 4. 获取所有A股（不过滤）
    # all_stocks = get_stock_pool(prefixes=[])  # 或 prefixes=None
    #
    # # 5. 获取沪市所有股票
    # sh_stocks = get_stock_pool(prefixes=['sh.'])