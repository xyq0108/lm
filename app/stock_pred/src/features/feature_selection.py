"""feature_selection — 特征选择工具"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from src.common.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """特征选择器 — 过滤低方差、高相关、低重要性特征"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.method = self.config.get("method", "variance")
        self.variance_threshold = self.config.get("variance_threshold", 1e-5)
        self.corr_threshold = self.config.get("corr_threshold", 0.95)

    def select(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """执行特征选择"""
        cols = list(feature_cols)
        n0 = len(cols)

        if self.method in ("variance", "all"):
            # 删除近零方差
            valid = []
            for c in cols:
                if df[c].std() > self.variance_threshold:
                    valid.append(c)
            cols = valid
            logger.info(f"低方差过滤: {n0} -> {len(cols)}")

        if self.method in ("correlation", "all"):
            # 删除高相关特征
            corr_matrix = df[cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
            cols = [c for c in cols if c not in to_drop]
            logger.info(f"高相关过滤: 删除 {len(to_drop)} 个特征，剩余 {len(cols)}")

        return df, cols


def get_feature_groups(feature_cols: List[str]) -> dict:
    """按前缀将特征分组，用于报告分析"""
    groups = {
        "technical": [],
        "alpha": [],
        "cross_section": [],
        "market": [],
        "other": [],
    }
    prefix_map = {
        "ma_": "technical", "std_": "technical", "ret_": "technical",
        "boll_": "technical", "rsi_": "technical", "close_": "technical",
        "amplitude": "technical",
        "corr_": "alpha", "money_": "alpha", "price_": "alpha", "mf_": "alpha",
        "_rank": "cross_section", "_zscore": "cross_section",
        "market_": "market", "excess_": "market",
    }
    for col in feature_cols:
        assigned = False
        for prefix, group in prefix_map.items():
            if col.startswith(prefix) or col.endswith(prefix):
                groups[group].append(col)
                assigned = True
                break
        if not assigned:
            groups["other"].append(col)
    return groups
