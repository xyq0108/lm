"""cross_section_features — 横截面特征（截面标准化、相对排名等）"""
import pandas as pd
import numpy as np
from src.features.base_features import BaseFeatureGenerator
from src.common.logger import get_logger

logger = get_logger(__name__)


class CrossSectionFeatures(BaseFeatureGenerator):
    """横截面特征：在每个交易日对全体股票做标准化/排名"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.rank_pct_cols = self.config.get("rank_pct_cols", ["涨跌幅", "换手率"])

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.rank_pct_cols:
            if col not in df.columns:
                continue
            # 截面排名 (0~1)
            rank_col = f"{col}_rank"
            df[rank_col] = df.groupby("日期")[col].rank(pct=True)

            # 截面 Z-score
            z_col = f"{col}_zscore"
            df[z_col] = df.groupby("日期")[col].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-8)
            )

        logger.info(f"横截面特征: 新增 {len(self.rank_pct_cols) * 2} 个特征")
        return df
