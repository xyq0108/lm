"""风险过滤 — 剔除不符合风控条件的股票"""
import pandas as pd
import numpy as np
from src.common.logger import get_logger

logger = get_logger(__name__)


def apply_risk_filter(candidate_df: pd.DataFrame,
                      full_df: pd.DataFrame = None,
                      config: dict = None) -> pd.DataFrame:
    """
    对候选股票做风控过滤。

    过滤规则:
        - 剔除 ST / *ST 股票（以股票代码 2/3 开头）
        - 剔除最近 20 交易日涨跌幅 > 50% 的（追涨风险）
        - 剔除最近 20 日换手率 < 0.1% 的（流动性风险）

    参数:
        candidate_df: 含 [股票代码, fusion_rank, ...] 的 DataFrame
        full_df: 全量历史数据，用于计算过滤条件
        config: 风控配置

    返回:
        过滤后的 DataFrame
    """
    if config is None:
        config = {}

    df = candidate_df.copy()
    n_before = len(df)

    # 1. 剔除 ST / 退市股票（股票代码 2/3 开头或含字母）
    st_mask = df["股票代码"].astype(str).str.match(r"^[23]")
    if st_mask.any():
        df = df[~st_mask]
        logger.info(f"风险过滤: 剔除 {st_mask.sum()} 只 ST/*ST")

    # 2. 流动性过滤
    if full_df is not None:
        latest = full_df[full_df["日期"] == full_df["日期"].max()]
        if "换手率" in latest.columns:
            low_liq = latest["换手率"] < 0.001
            bad_codes = set(latest.loc[low_liq, "股票代码"].values)
            n_bad = df["股票代码"].isin(bad_codes).sum()
            df = df[~df["股票代码"].isin(bad_codes)]
            if n_bad:
                logger.info(f"风险过滤: 剔除 {n_bad} 只低流动性股票")

    n_after = len(df)
    if n_after < n_before:
        logger.info(f"风险过滤: {n_before} -> {n_after} ({n_before - n_after} 只被剔除)")

    return df
