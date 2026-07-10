"""Score Fusion — 直接对预测分数做加权融合（不转排名）"""
import pandas as pd
import numpy as np
from typing import List, Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


def score_weighted_average(predictions: List[pd.DataFrame],
                           weights: List[float] = None,
                           normalize: bool = True) -> pd.DataFrame:
    """
    分数加权平均融合。

    参数:
        predictions: 每个元素含 [股票代码, score]
        weights: 各模型权重，None 表示等权
        normalize: 是否对每个模型的分数做 Z-score 归一化

    返回:
        含 [股票代码, fusion_score] 的 DataFrame
    """
    n_models = len(predictions)
    if n_models == 0:
        raise ValueError("预测列表为空")

    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        weights = np.array(weights) / sum(weights)

    all_stocks = set()
    for pred in predictions:
        all_stocks.update(pred["股票代码"].values)
    all_stocks = sorted(all_stocks)

    fusion_scores = {s: 0.0 for s in all_stocks}

    for pred, w in zip(predictions, weights):
        pred = pred.copy()
        scores = pred["score"].values
        if normalize and scores.std() > 1e-8:
            scores = (scores - scores.mean()) / scores.std()
        score_map = dict(zip(pred["股票代码"], scores))
        for stock in all_stocks:
            fusion_scores[stock] += w * score_map.get(stock, 0.0)

    result = pd.DataFrame({
        "股票代码": list(fusion_scores.keys()),
        "fusion_score": list(fusion_scores.values()),
    })
    result = result.sort_values("fusion_score", ascending=False).reset_index(drop=True)

    logger.info(f"Score 融合完成: {n_models} 个模型, {len(result)} 只候选股票")
    return result
