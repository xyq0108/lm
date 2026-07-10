"""模型选择 — 决定哪些模型进入最终融合"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


def select_models_for_ensemble(model_results: List[Dict],
                                min_mean_return: float = -0.01,
                                min_win_rate: float = 0.4,
                                max_correlation: float = 0.9) -> List[int]:
    """
    选择进入融合的模型。

    选择标准:
        1. 验证收益稳定
        2. 多窗口胜过 baseline
        3. 和其他模型相关性不要太高
        4. 最差窗口不要太差

    参数:
        model_results: 每个模型的验证结果字典列表
            [{"model_id": 0, "mean_return": 0.05, "std_return": 0.02,
              "win_rate": 0.6, "worst_return": -0.05, "daily_returns": Series}]
        min_mean_return: 最低平均收益
        min_win_rate: 最低胜率
        max_correlation: 最大允许相关性

    返回:
        选中的模型索引列表
    """
    n_models = len(model_results)
    if n_models == 0:
        return []

    # 初步过滤
    candidates = []
    for i, res in enumerate(model_results):
        mean_r = res.get("mean_return", -999)
        win_r = res.get("win_rate", 0)

        if mean_r >= min_mean_return and win_r >= min_win_rate:
            # Sharpe-like 比率 (均值/标准差)
            sharpe = mean_r / max(res.get("std_return", 0.001), 0.001)
            candidates.append({
                "idx": i,
                "sharpe": sharpe,
                "mean_return": mean_r,
                "win_rate": win_r,
                "worst_return": res.get("worst_return", -999),
            })

    if len(candidates) == 0:
        logger.warning("没有模型满足最低筛选条件，使用所有模型")
        return list(range(n_models))

    # 按 Sharpe 排序
    candidates.sort(key=lambda x: x["sharpe"], reverse=True)

    # 检查相关性（如果有收益率序列）
    if "daily_returns" in model_results[0]:
        selected = [candidates[0]["idx"]]
        for c in candidates[1:]:
            # 与已选模型的相关性
            corrs = []
            for sel_idx in selected:
                sr1 = model_results[c["idx"]]["daily_returns"]
                sr2 = model_results[sel_idx]["daily_returns"]
                common = sr1.index.intersection(sr2.index)
                if len(common) > 5:
                    corr = sr1.loc[common].corr(sr2.loc[common])
                    corrs.append(corr)
            # 如果与所有已选模型的相关性都低于阈值，则加入
            if all(c <= max_correlation for c in corrs):
                selected.append(c["idx"])
        return selected
    else:
        # 无收益率序列时，取 Top-3
        return [c["idx"] for c in candidates[:3]]


def calculate_model_correlation(model_results: List[Dict]) -> pd.DataFrame:
    """计算各模型每日收益的相关性矩阵"""
    returns_dict = {}
    for i, res in enumerate(model_results):
        if "daily_returns" in res:
            returns_dict[f"model_{i}"] = res["daily_returns"]

    if len(returns_dict) < 2:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict)
    return returns_df.corr()
