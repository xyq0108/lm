"""OHLCV 数据清洗"""
import pandas as pd
import numpy as np
from src.common.logger import get_logger

logger = get_logger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 OHLCV 数据：
    - 删除价格或成交量为空/负值的行
    - 删除开盘/收盘/最高/最低有异常的行
    - 按 (股票代码, 日期) 去重
    - 填充少量缺失的涨跌幅为 0
    """
    df = df.copy()
    n0 = len(df)

    # 删除价格列为空的记录
    price_cols = ["开盘", "收盘", "最高", "最低"]
    df = df.dropna(subset=price_cols + ["成交量"])
    n1 = len(df)
    if n1 < n0:
        logger.info(f"删除 {n0 - n1} 行空值价格/成交量记录")

    # 删除价格 <= 0 的记录
    for col in price_cols:
        df = df[df[col] > 0]
    df = df[df["成交量"] >= 0]
    n2 = len(df)
    if n2 < n1:
        logger.info(f"删除 {n1 - n2} 行无效价格/成交量记录")

    # 修复最高最低关系
    df["最高"] = df[["开盘", "收盘", "最高"]].max(axis=1)
    df["最低"] = df[["开盘", "收盘", "最低"]].min(axis=1)

    # 按 (股票代码, 日期) 去重，保留最后一条
    df = df.sort_values(["股票代码", "日期"]).drop_duplicates(
        subset=["股票代码", "日期"], keep="last"
    ).reset_index(drop=True)

    # 填充涨跌幅缺失为 0
    if "涨跌幅" in df.columns:
        df["涨跌幅"] = df["涨跌幅"].fillna(0.0)

    logger.info(f"清洗完成: {n0} -> {len(df)} 行, "
                f"删除 {n0 - len(df)} 行 ({(n0 - len(df)) / max(n0, 1):.1%})")
    return df
