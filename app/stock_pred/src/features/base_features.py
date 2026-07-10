"""base_features — 特征生成的抽象基类"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class BaseFeatureGenerator(ABC):
    """特征生成器基类，每个子类实现一种特征群"""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """从数据中学习（如滚动窗口参数）"""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成特征并拼接到 df"""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
