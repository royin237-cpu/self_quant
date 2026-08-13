import pandas as pd
import numpy as np
import os
import glob
import re
from tqdm import tqdm

# ================= 路径配置 =================
DATA_PATH = r'F:\self_quant\data\data'

# 正常票和退市票的数据文件夹映射
DIRS = [
    {
        'type': '正常票',
        'kline': os.path.join(DATA_PATH, '日K线'),
        'profit': os.path.join(DATA_PATH, '利润'),
        'balance': os.path.join(DATA_PATH, '资产负债'),
        'cashflow': os.path.join(DATA_PATH, '现金流量'),
    },
    {
        'type': '退市票',
        'kline': os.path.join(DATA_PATH, '退市票日K线'),
        'profit': os.path.join(DATA_PATH, '退市票利润'),
        'balance': os.path.join(DATA_PATH, '退市票资产负债'),
        'cashflow': os.path.join(DATA_PATH, '退市票现金流量'),
    }
]

# ================= 列名配置 =================
# 你可以根据需要删减字段
income_columns = [
    '报告日', '净利润', '营业总收入', '营业总成本', '营业成本',
    '营业利润', '利润总额', '所得税费用', '归属于母公司所有者的净利润'
]

balance_columns = [
    '报告日', '货币资金', '应收账款', '存货', '流动资产合计',
    '非流动资产合计', '资产总计', '流动负债合计', '非流动负债合计',
    '负债合计', '所有者权益(或股东权益)合计', '归属于母公司所有者权益(或股东权益)合计'
]

cashflow_columns = [
    '报告日', '经营活动产生的现金流量净额', '投资活动产生的现金流量净额',
    '筹资活动产生的现金流量净额', '现金及现金等价物净增加额'
]

# ================= 核心功能函数 =================
def load_financial_statement(file_path, columns):
    """读取单个财务表并规范化格式，处理编码问题"""
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame(columns=columns)

    # 兼容处理GBK和UTF-8编码，避免表头乱码
    try:
        df_full = pd.read_csv(file_path, encoding='gbk')
    except UnicodeDecodeError:
        try:
            df_full = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"读取报错 {file_path}: {e}")
            return pd.DataFrame(columns=columns)

    valid_columns = [col for col in columns if col in df_full.columns]
    df = df_full[valid_columns].copy()

    if '报告日' in df.columns:
        # 将格式类似 20241231 转换为日期
        df['报告日'] = pd.to_datetime(df['报告日'], format='%Y%m%d', errors='coerce')
        # 去除没有报告日期的空行
        df = df.dropna(subset=['报告日'])

    return df

def find_file_by_code(directory, code_6digit):
    """根据6位股票代码在指定文件夹中模糊查找文件"""
    search_pattern = os.path.join(directory, f"*{code_6digit}*.csv")
    files = glob.glob(search_pattern)
    return files[0] if len(files) > 0 else None

def process_single_stock(kline_file, profit_dir, balance_dir, cashflow_dir):
    """处理并合并单只股票的所有数据，防止未来函数"""
    filename = os.path.basename(kline_file)
    match = re.search(r'\d{6}', filename)
    if not match:
        return None

    code_6digit = match.group()

    # 1. 读取K线数据
    try:
        kline_df = pd.read_csv(kline_file, encoding='gbk')
    except UnicodeDecodeError:
        kline_df = pd.read_csv(kline_file, encoding='utf-8')

    if 'date' not in kline_df.columns:
        return None

    # K线日期格式化并排序（必须排序才能使用 merge_asof）
    kline_df['date'] = pd.to_datetime(kline_df['date'])
    kline_df = kline_df.sort_values('date')

    # 2. 寻找对应的财务报表文件路径
    profit_file = find_file_by_code(profit_dir, code_6digit)
    balance_file = find_file_by_code(balance_dir, code_6digit)
    cashflow_file = find_file_by_code(cashflow_dir, code_6digit)

    # 3. 加载三大表
    profit_df = load_financial_statement(profit_file, income_columns)
    balance_df = load_financial_statement(balance_file, balance_columns)
    cashflow_df = load_financial_statement(cashflow_file, cashflow_columns)

    # 4. 合并三大财报（根据'报告日' outer join）
    fin_df = pd.DataFrame(columns=['报告日'])
    if not profit_df.empty:
        fin_df = pd.merge(fin_df, profit_df, on='报告日', how='outer')
    if not balance_df.empty:
        fin_df = pd.merge(fin_df, balance_df, on='报告日', how='outer')
    if not cashflow_df.empty:
        fin_df = pd.merge(fin_df, cashflow_df, on='报告日', how='outer')

    if fin_df.empty or len(fin_df.columns) == 1:
        return kline_df

    # 5. 核心逻辑：避免未来函数
    # 假设报告日期的披露存在滞后性，一般向后推3个月（使用 pd.DateOffset）
    fin_df['实际披露日'] = fin_df['报告日'] + pd.DateOffset(months=3)

    # 清理并排序（必须按照合并键排序）
    fin_df = fin_df.dropna(subset=['实际披露日'])
    fin_df = fin_df.sort_values('实际披露日')

    # 6. 拼接到日K线
    # 使用 merge_asof: 针对K线的每一个 'date'，向后寻找距离它最近且 '实际披露日' <= 'date' 的财务数据
    # direction='backward' 意味着使用“历史最新可用数据”
    merged_df = pd.merge_asof(
        kline_df,
        fin_df,
        left_on='date',
        right_on='实际披露日',
        direction='backward'
    )

    return merged_df

# ================= 主程序：批量处理 =================
if __name__ == '__main__':
    all_data_list = []
    
    for dir_info in DIRS:
        stock_type = dir_info['type']
        kline_dir = dir_info['kline']
        profit_dir = dir_info['profit']
        balance_dir = dir_info['balance']
        cashflow_dir = dir_info['cashflow']
        
        kline_files = glob.glob(os.path.join(kline_dir, "*.csv"))
        print(f"\n[{stock_type}] 共找到 {len(kline_files)} 个K线文件，开始处理并合并三大表...")
        
        for kline_file in tqdm(kline_files, desc=stock_type):
            merged_df = process_single_stock(kline_file, profit_dir, balance_dir, cashflow_dir)
            
            if merged_df is not None and not merged_df.empty:
                # 增加标识区分正常和退市股票 (可选)
                merged_df['stock_status'] = stock_type
                all_data_list.append(merged_df)
                
    if all_data_list:
        print("\n正在将所有股票数据拼接成一个大表...")
        final_all_data = pd.concat(all_data_list, ignore_index=True)
        
        # 按照日期和代码进行排序，标准的面板数据格式
        print("正在按代码和日期排序并进行退市整理期清洗...")
        final_all_data = final_all_data.sort_values(by=['code', 'date'])
        final_all_data.reset_index(drop=True, inplace=True)
        final_all_data['original_idx'] = final_all_data.index

        # ==========================================
        # 修复退市整理期股票状态位
        # ==========================================
        print("识别并清理退市整理期数据...")
        global_max_date = final_all_data['date'].max()
        max_dates = final_all_data.groupby('code')['date'].max()
        delisted_codes = max_dates[max_dates < global_max_date].index

        delisted_mask = final_all_data['code'].isin(delisted_codes)

        if 'tradestatus' in final_all_data.columns:
            # 找到每只退市股最后一次停牌 (tradestatus == 0) 的绝对行索引
            last_zero_indices = final_all_data[delisted_mask & (final_all_data['tradestatus'] == 0)].groupby('code')['original_idx'].max()

            # 找到每只退市股最后一天的绝对行索引
            last_day_indices = final_all_data[delisted_mask].groupby('code')['original_idx'].max()

            compare_df = pd.DataFrame({
                'last_zero_idx': last_zero_indices,
                'last_day_idx': last_day_indices
            }).dropna()

            # 计算最后一次停牌到最终彻底退市，中间隔了多少个交易日（行数）
            compare_df['days_diff'] = compare_df['last_day_idx'] - compare_df['last_zero_idx']

            # 放宽到100天足以涵盖各种异常停复牌
            target_codes = compare_df[(compare_df['days_diff'] > 0) & (compare_df['days_diff'] <= 100)]

            indices_to_fix = []
            for code, row in target_codes.iterrows():
                idx_start = int(row['last_zero_idx']) + 1
                idx_end = int(row['last_day_idx'])
                indices_to_fix.extend(list(range(idx_start, idx_end + 1)))

            if indices_to_fix:
                final_all_data.loc[indices_to_fix, 'tradestatus'] = 0
                if 'isST' in final_all_data.columns:
                    final_all_data.loc[indices_to_fix, 'isST'] = 1
                print(f"已成功将 {len(indices_to_fix)} 条退市整理期数据的 tradestatus 抹平为 0。")

            print("正在执行兜底清理，拉黑所有退市股最后 25 个交易日...")
            fallback_indices = final_all_data[delisted_mask].groupby('code').tail(25).index
            final_all_data.loc[fallback_indices, 'tradestatus'] = 0
            if 'isST' in final_all_data.columns:
                final_all_data.loc[fallback_indices, 'isST'] = 1

        final_all_data.drop(columns=['original_idx'], inplace=True)
        
        # 计算流通市值和流通股本
        print("正在计算 流通股本 和 流通市值...")
        if 'volume' in final_all_data.columns and 'turn' in final_all_data.columns and 'close' in final_all_data.columns:
            # 避免除以0报错，将换手率为0的替换为空值(NaN)
            final_all_data['turn'] = final_all_data['turn'].replace(0, np.nan)
            # 换算逻辑：成交量 / (换手率 / 100)
            final_all_data['流通股本'] = final_all_data['volume'] / (final_all_data['turn'] / 100)
            final_all_data['流通市值'] = final_all_data['流通股本'] * final_all_data['close']
        else:
            print("警告：缺少 volume, turn 或 close 列，无法计算流通市值。")

        # 恢复成按日期和代码排序，符合回测引擎读取需求
        final_all_data = final_all_data.sort_values(by=['date', 'code'])

        # 强制去重，保证 date 和 code 的联合主键唯一性
        print("正在执行严格去重检查 (基于 date 和 code)...")
        initial_len = len(final_all_data)
        final_all_data.drop_duplicates(subset=['date', 'code'], keep='last', inplace=True)
        dedup_len = len(final_all_data)
        if initial_len != dedup_len:
            print(f"⚠️ 发现并清理了 {initial_len - dedup_len} 条重复数据记录。")

        # 1. 推荐：保存为 Parquet 文件（体积小、读写极快、且天然支持避免乱码）
        output_parquet = os.path.join(DATA_PATH, "all_stocks_merged_fixed.parquet")
        print(f"正在保存为 Parquet 格式: {output_parquet}")
        final_all_data.to_parquet(output_parquet, engine='pyarrow', index=False)
        
        # 2. 如果你需要用 Excel 打开看，可以导出一个前1000行的CSV样例供核对表头
        sample_csv = os.path.join(DATA_PATH, "full_market_merged_sample.csv")
        # 使用 utf-8-sig 编码，这样在 Windows 下用 Excel 打开不会乱码
        final_all_data.head(1000).to_csv(sample_csv, index=False, encoding='utf-8-sig')
        
        print("\n太棒了！合并全部完成。")
        print(f"数据总行数: {len(final_all_data)} 行")
        print(f"大表 Parquet 路径 (用于量化回测): {output_parquet}")
        print(f"表头核对 CSV 路径 (用于 Excel 预览): {sample_csv}")
    else:
        print("未获取到任何数据！")