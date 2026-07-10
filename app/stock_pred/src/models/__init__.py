"""模型基类 — 定义所有模型的统一接口"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """所有模型的抽象基类"""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
            feature_cols: list) -> None:
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...
