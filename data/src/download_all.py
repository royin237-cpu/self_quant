import baostock as bs
import os
import time
from tqdm import tqdm  # 1. 导入 tqdm
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError # 引入完整性错误类
from download_onecode import get_stock_data
from stock_pool import get_stock_pool
from sqlalchemy.types import String, DateTime

# 数据库连接
DB_URI = "mysql+pymysql://vnpy:Ta1zheng21@localhost:3306/vnpy?charset=utf8"
engine = create_engine(DB_URI)


def save_data(df, code, save_path):
    # 保存 CSV (CSV 本身没有唯一约束，这里逻辑由你决定是否覆盖)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, f"{code}.csv"), index=False)

    # 尝试存入 MySQL
    try:
        df.to_sql('a_stock_analysis', engine, if_exists='append', index=False,
                  dtype={
                          'code': String(20),
                          'date': DateTime()
                        })
        print(f"股票 {code} 数据已存入。")
    except IntegrityError:
        # 如果捕获到唯一键冲突，说明数据已存在，直接忽略
        print(f"股票 {code} 数据已存在，跳过。")
    except Exception as e:
        print(f"存入 {code} 时发生其他错误: {e}")


if __name__ == "__main__":
    bs.login()

    IS_BATCH = True  # 改为 True 进行批量下载
    start_date = "2010-01-01"
    end_date = "2026-03-20"

    if IS_BATCH:
        codes = get_stock_pool()

        # 2. 使用 tqdm 包裹你的循环
        # desc 是进度条描述，unit 是计数单位
        for code in tqdm(codes, desc="正在下载全市场股票数据", unit="stock"):
            try:
                # # 稍微加点随机间隔，避免对服务器造成瞬时压力
                # time.sleep(0.1)

                df = get_stock_data(code, start_date, end_date)
                if df is not None and not df.empty:
                    save_data(df, code, r"F:\self_quant\data\data")
            except Exception as e:
                # 使用 tqdm.write 来输出错误，这样不会破坏进度条的样式
                tqdm.write(f"\n下载 {code} 出错: {e}")
                # 简单重连机制
                bs.logout()
                bs.login()
    else:
        code = "sh.600519"
        df = get_stock_data(code, start_date, end_date)
        save_data(df, code, r"F:\self_quant\data\data")

    bs.logout()