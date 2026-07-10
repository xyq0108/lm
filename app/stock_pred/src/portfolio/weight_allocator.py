"""权重分配 — 为选中的股票分配投资权重"""
import pandas as pd
import numpy as np
from src.common.logger import get_logger

logger = get_logger(__name__)


def allocate_weights(selected: pd.DataFrame,
                     method: str = "equal",
                     max_single_weight: float = 0.3,
                     score_col: str = "score") -> pd.DataFrame:
    """
    为选中的股票分配权重。

    参数:
        selected: 含 [stock_id, score] 的 DataFrame
        method: "equal" | "score_weighted" | "rank_weighted"
        max_single_weight: 单只最大权重
        score_col: 分数列名

    返回:
        含 [stock_id, weight] 的 DataFrame (权重和为 1)
    """
    result = selected[["stock_id"]].copy()
    n = len(result)

    if n == 0:
        logger.error("选股列表为空，无法分配权重")
        return pd.DataFrame(columns=["stock_id", "weight"])

    if method == "equal":
        weights = np.ones(n) / n

    elif method == "score_weighted":
        scores = selected[score_col].values.copy()
        scores = np.maximum(scores, 0)  # 负分置 0
        if scores.sum() == 0:
            scores = np.ones(n)
        weights = scores / scores.sum()

    elif method == "rank_weighted":
        # 线性递减权重: 第1名 n/(1+...+n), 第2名 (n-1)/(1+...+n)
        ranks = np.arange(1, n + 1, dtype=float)
        weights = (n - ranks + 1) / ((n * (n + 1)) / 2)

    else:
        logger.warning(f"未知权重方法 '{method}'，使用等权")
        weights = np.ones(n) / n

    # 单只上限截断，重新归一化
    weights = np.minimum(weights, max_single_weight)
    weights = weights / weights.sum()

    result["weight"] = np.round(weights, 6)
    logger.info(f"权重分配: {method}, {n} 只股票, 最大 {max_single_weight}")
    return result
