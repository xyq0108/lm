"""technical_features — 技术指标特征（MA, RSI, MACD, Bollinger 等）"""
import pandas as pd
import numpy as np
from src.features.base_features import BaseFeatureGenerator
from src.common.logger import get_logger

logger = get_logger(__name__)


class TechnicalFeatures(BaseFeatureGenerator):
    """技术指标特征生成器"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.windows = self.config.get("windows", [5, 10, 20, 60])

    def fit(self, df: pd.DataFrame) -> None:
        pass  # 无需拟合参数

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["股票代码", "日期"])
        result = []

        for code, group in df.groupby("股票代码"):
            g = group.copy()
            close = g["收盘"]
            high = g["最高"]
            low = g["最低"]
            volume = g["成交量"]

            for w in self.windows:
                g[f"ma_{w}"] = close.rolling(w, min_periods=1).mean()
                g[f"std_{w}"] = close.rolling(w, min_periods=1).std()
                g[f"ret_{w}d"] = close.pct_change(w).fillna(0)
                g[f"volume_ma_{w}"] = volume.rolling(w, min_periods=1).mean()

                # Bollinger %B
                mid = g[f"ma_{w}"]
                std = g[f"std_{w}"]
                g[f"boll_{w}"] = (close - mid) / (2 * std + 1e-8)

            # RSI 14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            g["rsi_14"] = 100 - 100 / (1 + rs)

            # 价格位置
            g["close_high_ratio"] = (close - low.rolling(20, min_periods=1).min()) / \
                (high.rolling(20, min_periods=1).max() - low.rolling(20, min_periods=1).min() + 1e-8)

            result.append(g)

        out = pd.concat(result).sort_index()
        new_cols = [c for c in out.columns if c not in df.columns]
        logger.info(f"技术指标: 新增 {len(new_cols)} 个特征")
        return out
