"""Baseline 对比 — 将模型表现与基准策略比较"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


def compare_with_baseline(model_returns: pd.Series,
                          baseline_path: Optional[str] = None,
                          baseline_returns: Optional[pd.Series] = None) -> dict:
    """比较模型 vs Baseline 的收益表现"""
    if baseline_returns is None and baseline_path:
        try:
            baseline_df = pd.read_csv(baseline_path)
            baseline_returns = baseline_df.iloc[:, 0]
        except Exception as e:
            logger.warning(f"读取 baseline 失败: {e}，使用 0 收益")
            baseline_returns = pd.Series(0, index=model_returns.index)

    if baseline_returns is None:
        baseline_returns = pd.Series(0, index=model_returns.index)

    # 对齐索引
    common = model_returns.index.intersection(baseline_returns.index)
    mr = model_returns.loc[common]
    br = baseline_returns.loc[common]

    win_rate = (mr > br).mean()
    avg_excess = (mr - br).mean()
    t_stat = (mr - br).mean() / (mr - br).std() * np.sqrt(len(common)) if (mr - br).std() > 0 else 0

    result = {
        "model_mean_return": float(mr.mean()),
        "baseline_mean_return": float(br.mean()),
        "win_rate": float(win_rate),
        "avg_excess_return": float(avg_excess),
        "t_statistic": float(t_stat),
        "n_days": len(common),
    }
    logger.info(f"Baseline 对比: 胜率 {win_rate:.2%}, 超额 {avg_excess:.4%}")
    return result
