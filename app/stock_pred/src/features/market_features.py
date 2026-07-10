"""market_features — 市场整体特征（大盘收益、行业均值等）"""
import pandas as pd
import numpy as np
from src.features.base_features import BaseFeatureGenerator
from src.common.logger import get_logger

logger = get_logger(__name__)


class MarketFeatures(BaseFeatureGenerator):
    """市场宏观特征 — 全市场等权/加权平均值"""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 每日市场平均收益
        market_ret = df.groupby("日期")["涨跌幅"].mean().rename("market_ret")
        market_vol = df.groupby("日期")["换手率"].mean().rename("market_turnover")

        df = df.merge(market_ret, on="日期", how="left")
        df = df.merge(market_vol, on="日期", how="left")

        # 相对市场超额
        df["excess_ret"] = df["涨跌幅"] - df["market_ret"]

        # 市场波动率 (20日滚动)
        daily_vol = df.groupby("日期")["涨跌幅"].std().rolling(20).mean().rename("market_volatility")
        df = df.merge(daily_vol, on="日期", how="left")
        df["market_volatility"] = df.groupby("股票代码")["market_volatility"].ffill().fillna(0)

        logger.info(f"市场特征: 新增 market_ret, market_turnover, excess_ret, market_volatility")
        return df
