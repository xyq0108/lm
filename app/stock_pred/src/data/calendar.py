"""交易日历逻辑 — T+1 和 T+5 是交易日，不是自然日"""
import pandas as pd
import numpy as np
from typing import List
from src.common.logger import get_logger

logger = get_logger(__name__)


def get_trading_calendar(df: pd.DataFrame) -> List[pd.Timestamp]:
    """从数据中提取所有交易日序列（去重排序后的日期列表）"""
    trading_days = sorted(df["日期"].unique())
    return trading_days


def add_trading_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    给每个交易日编号，使得 T+1 和 T+5 基于交易日偏移而非自然日偏移。
    每个日期映射到一个交易日序号（Tidx），Tidx 从 0 开始递增。
    """
    trading_days = get_trading_calendar(df)
    date_to_idx = {d: i for i, d in enumerate(trading_days)}
    df = df.copy()
    df["trading_idx"] = df["日期"].map(date_to_idx)
    df = df.sort_values(["股票代码", "trading_idx"]).reset_index(drop=True)

    logger.info(f"交易日历构建完成: 共 {len(trading_days)} 个交易日, "
                f"范围 {pd.Timestamp(trading_days[0]).date()} ~ {pd.Timestamp(trading_days[-1]).date()}")
    return df


def get_future_date(df: pd.DataFrame, current_date: pd.Timestamp, shift: int) -> pd.Timestamp:
    """获取某个日期往后 shift 个交易日的日期（基于当前数据的日历）"""
    trading_days = get_trading_calendar(df)
    date_to_idx = {d: i for i, d in enumerate(trading_days)}
    current_idx = date_to_idx.get(current_date)
    if current_idx is None:
        return pd.NaT
    target_idx = current_idx + shift
    if target_idx < 0 or target_idx >= len(trading_days):
        return pd.NaT
    return trading_days[target_idx]
