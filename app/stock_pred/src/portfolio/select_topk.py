"""Top-K 选股 — 从预测分数中选择最终股票"""
import pandas as pd
import numpy as np
from typing import Optional, List
from src.common.logger import get_logger
from src.common.checks import validate_result_csv

logger = get_logger(__name__)


def select_topk_stocks(pred_df: pd.DataFrame,
                       pred_score_col: str = "fusion_rank",
                       stock_col: str = "股票代码",
                       top_k: int = 5,
                       allow_less: bool = True) -> pd.DataFrame:
    """
    按最终分数选择前 top_k 只股票。

    筛选前检查:
        - 股票代码是否有效
        - 是否预测分数为 NaN
        - 是否有足够历史数据

    返回:
        包含 stock_id 和 score 的 DataFrame
    """
    df = pred_df.copy()

    # 过滤无效数据
    n_before = len(df)
    df = df[df[pred_score_col].notna()].copy()
    if len(df) < n_before:
        logger.warning(f"过滤了 {n_before - len(df)} 条分数为 NaN 的股票")

    # 按分数排序取 Top-K
    df = df.sort_values(pred_score_col, ascending=False)

    if allow_less or len(df) >= top_k:
        selected = df.head(top_k).copy()
        n_selected = len(selected)
        if n_selected < top_k:
            logger.warning(f"只选出 {n_selected} 只股票（目标 {top_k}）")
    else:
        logger.error(f"可用股票 {len(df)} 不足 {top_k}")
        return pd.DataFrame(columns=["stock_id", "weight"])

    selected["stock_id"] = selected[stock_col].astype(str).str.zfill(6)
    selected["score"] = selected[pred_score_col]

    logger.info(f"选股完成: Top-{n_selected}/{(allow_less and '自适应' or str(top_k))}")
    return selected[["stock_id", "score"]]


def build_latest_candidates(df: pd.DataFrame, latest_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    为预测日构造候选股票集。

    在最新交易日，获取所有可交易股票的最新特征。
    """
    if latest_date is None:
        latest_date = df["日期"].max()

    latest = df[df["日期"] == latest_date].copy()
    logger.info(f"候选股票集: {latest_date.date()}, {len(latest)} 只股票")
    return latest
