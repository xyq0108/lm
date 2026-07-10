"""排序学习（Learning to Rank）相关性标签"""
import pandas as pd
import numpy as np
from src.common.logger import get_logger

logger = get_logger(__name__)


def add_all_relevance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每只股票在每个交易日构造排序学习的 relevance 标签。

    对同一交易日内的所有股票，按未来收益率分档打分：
    - top 20% → relevance = 3（强推荐）
    - top 20%~50% → relevance = 2（推荐）
    - 其余 → relevance = 1（中性）

    新增列:
        relevance: 排序学习相关性分数 (1/2/3)
    """
    df = df.copy()

    # 找未来收益率列
    ret_cols = [c for c in df.columns if c.startswith("future_return_")]
    if not ret_cols:
        logger.warning("未找到 future_return_* 列，跳过 relevance 标签")
        df["relevance"] = 1
        return df

    # 使用最长的 future_return 窗口（如 future_return_5d > future_return_1d）
    ret_col = max(ret_cols, key=lambda c: int(c.split("_")[-1].replace("d", "")))
    logger.info(f"使用 {ret_col} 构建 relevance 标签")

    # 按日期分组打分
    def _score_group(g: pd.DataFrame) -> pd.Series:
        vals = g[ret_col].dropna()
        if len(vals) < 5:
            return pd.Series(1, index=g.index)

        # 按收益率排序（降序）
        sorted_vals = vals.sort_values(ascending=False)
        n = len(sorted_vals)
        top20 = sorted_vals.iloc[:max(1, int(n * 0.2))]
        top50 = sorted_vals.iloc[:max(1, int(n * 0.5))]

        scores = pd.Series(1, index=g.index)
        scores.loc[top50.index] = 2
        scores.loc[top20.index] = 3
        return scores

    df["relevance"] = df.groupby("日期", group_keys=False, sort=False).apply(
        _score_group
    )

    # 对 NaN future_return 的行，relevance 置 1
    df.loc[df[ret_col].isna(), "relevance"] = 1
    df["relevance"] = df["relevance"].astype(int)

    logger.info(
        f"relevance 分布: "
        f"1={ (df['relevance'] == 1).sum() }, "
        f"2={ (df['relevance'] == 2).sum() }, "
        f"3={ (df['relevance'] == 3).sum() }"
    )
    return df
