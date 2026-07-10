"""Rank Fusion — 多个模型的排序分数融合"""
import pandas as pd
import numpy as np
from typing import List
from src.common.logger import get_logger

logger = get_logger(__name__)


def rank_average(predictions: List[pd.DataFrame], weights: List[float] = None) -> pd.DataFrame:
    """
    Rank Average 融合：将多个模型的预测分数转为排名后加权平均。

    参数:
        predictions: 每个元素是含 [股票代码, score] 的 DataFrame
        weights: 各模型权重，None 表示等权

    返回:
        含 [股票代码, fusion_rank] 的 DataFrame
    """
    n_models = len(predictions)
    if n_models == 0:
        raise ValueError("预测列表为空")

    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        weights = np.array(weights) / sum(weights)

    # 收集所有候选股票
    all_stocks = set()
    for pred in predictions:
        all_stocks.update(pred["股票代码"].values)
    all_stocks = sorted(all_stocks)

    fusion_scores = {s: 0.0 for s in all_stocks}

    for pred, w in zip(predictions, weights):
        # 每个模型内排名归一化到 [0, 1]
        pred = pred.copy()
        pred["rank"] = pred["score"].rank(pct=True)
        rank_map = dict(zip(pred["股票代码"], pred["rank"]))
        for stock in all_stocks:
            fusion_scores[stock] += w * rank_map.get(stock, 0.0)

    result = pd.DataFrame({
        "股票代码": list(fusion_scores.keys()),
        "fusion_rank": list(fusion_scores.values()),
    })
    result = result.sort_values("fusion_rank", ascending=False).reset_index(drop=True)

    logger.info(f"Rank 融合完成: {n_models} 个模型, {len(result)} 只候选股票")
    return result
