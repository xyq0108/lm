"""未来收益标签构造"""
import pandas as pd
import numpy as np
from src.common.logger import get_logger

logger = get_logger(__name__)


def make_future_return_label(
    df: pd.DataFrame,
    buy_shift: int = 1,
    sell_shift: int = 5,
) -> pd.DataFrame:
    """
    构造未来收益标签（基于交易日偏移）。

    对每只股票，计算 sell_shift 个交易日后的收盘价相对当前收盘价的收益率。
    同时加入买入信号列（buy_shift 日后开盘相对当前收盘的涨跌幅）。

    新增列:
        future_return_{sell_shift}d: 未来 sell_shift 日收益率
        future_open_{buy_shift}d:    未来 buy_shift 日开盘相对涨跌幅
    """
    df = df.copy()

    # 确保按 (股票代码, trading_idx) 排序
    if "trading_idx" not in df.columns:
        raise ValueError("缺少 trading_idx 列，请先调用 add_trading_index()")

    df = df.sort_values(["股票代码", "trading_idx"]).reset_index(drop=True)

    # 未来收益列名
    ret_col = f"future_return_{sell_shift}d"
    open_col = f"future_open_{buy_shift}d"

    # 按股票分组，shift 得到未来价格
    df[ret_col] = (
        df.groupby("股票代码")["收盘"]
        .transform(lambda s: s.shift(-sell_shift))
    )
    df[open_col] = (
        df.groupby("股票代码")["开盘"]
        .transform(lambda s: s.shift(-buy_shift))
    )

    # 转为收益率
    df[ret_col] = (df[ret_col] - df["收盘"]) / df["收盘"]
    df[open_col] = (df[open_col] - df["收盘"]) / df["收盘"]

    # 填充分组后最后几行产生的 NaN（无未来数据）
    n_nan_ret = df[ret_col].isna().sum()
    n_nan_open = df[open_col].isna().sum()
    if n_nan_ret:
        logger.info(f"{ret_col}: {n_nan_ret} 行无未来数据 (NaN)")
    if n_nan_open:
        logger.info(f"{open_col}: {n_nan_open} 行无未来数据 (NaN)")

    return df
