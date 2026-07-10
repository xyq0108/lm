"""验证指标 — TopK 收益、NDCG、Hit Rate、Baseline 胜率"""
import numpy as np
import pandas as pd


def topk_return(pred_df: pd.DataFrame, label_col: str = "future_return_5d",
                score_col: str = "score", top_k: int = 5) -> float:
    """Top-K 等权组合收益"""
    if len(pred_df) < top_k:
        return 0.0
    top = pred_df.nlargest(top_k, score_col)
    return float(top[label_col].mean())


def ndcg_at_k(pred_df: pd.DataFrame, label_col: str = "future_return_5d",
              score_col: str = "score", k: int = 5) -> float:
    """NDCG@K — 归一化折损累计收益"""
    if len(pred_df) < k:
        return 0.0

    # 按 label 排序为 ideal
    ideal = pred_df.nlargest(k, label_col)[label_col].values
    dcg_ideal = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))

    ranked = pred_df.nlargest(k, score_col)[label_col].values
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ranked))

    return dcg / dcg_ideal if dcg_ideal > 0 else 0.0


def hit_rate_at_k(pred_df: pd.DataFrame, label_col: str = "future_return_5d",
                  score_col: str = "score", k: int = 5, top_pct: float = 0.2) -> float:
    """Hit-Rate@K: Top-K 选股中落入真实 Top-20% 的比例"""
    n_top = max(1, int(len(pred_df) * top_pct))
    true_top = set(pred_df.nlargest(n_top, label_col).index)
    pred_top = set(pred_df.nlargest(k, score_col).index)
    hits = len(true_top & pred_top)
    return hits / k


def baseline_win_rate(daily_returns: pd.Series,
                      baseline_returns: pd.Series = None) -> float:
    """模型日收益超 Baseline 的比例"""
    if baseline_returns is None:
        baseline_returns = pd.Series(0, index=daily_returns.index)
    aligned = pd.concat([daily_returns, baseline_returns], axis=1).dropna()
    if len(aligned) == 0:
        return 0.0
    return float((aligned.iloc[:, 0] > aligned.iloc[:, 1]).mean())
