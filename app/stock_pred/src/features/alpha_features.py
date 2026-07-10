"""alpha_features — Alpha 因子特征（量价关系、资金流等）"""
import pandas as pd
import numpy as np
from src.features.base_features import BaseFeatureGenerator
from src.common.logger import get_logger

logger = get_logger(__name__)


class AlphaFeatures(BaseFeatureGenerator):
    """Alpha 因子特征生成"""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["股票代码", "日期"])
        result = []

        for code, group in df.groupby("股票代码"):
            g = group.copy()
            close = g["收盘"]
            volume = g["成交量"]
            amount = g["成交额"] if "成交额" in g.columns else close * volume
            vwap = amount / (volume + 1e-8)

            # 量价相关性 (20日滚动)
            g["corr_vol_price_20"] = close.rolling(20).corr(volume).fillna(0)

            # 资金流强度
            g["money_flow"] = (close.diff() * volume).fillna(0)
            g["mf_ma_5"] = g["money_flow"].rolling(5, min_periods=1).mean()

            # 价格冲击
            g["price_impact"] = close.pct_change() / (volume / volume.rolling(5).mean() + 1e-8)
            g["price_impact"] = g["price_impact"].fillna(0)

            # 日内振幅
            g["amplitude"] = (g["最高"] - g["最低"]) / g["开盘"].replace(0, np.nan)
            g["amplitude"] = g["amplitude"].fillna(0)

            result.append(g)

        out = pd.concat(result).sort_index()
        new_cols = [c for c in out.columns if c not in df.columns]
        logger.info(f"Alpha因子: 新增 {len(new_cols)} 个特征")
        return out
