"""CatBoost 模型"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from src.models import BaseModel
from src.common.logger import get_logger

logger = get_logger(__name__)


class CatBoostModel(BaseModel):
    """CatBoost 回归模型"""

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.params.setdefault("iterations", 500)
        self.params.setdefault("learning_rate", 0.05)
        self.params.setdefault("depth", 6)
        self.params.setdefault("loss_function", "RMSE")
        self.params.setdefault("eval_metric", "RMSE")
        self.params.setdefault("verbose", False)
        self.params.setdefault("random_seed", 42)
        self.model = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
            feature_cols: list) -> None:
        train_y = train_df["future_return_5d"].values
        valid_y = valid_df["future_return_5d"].values

        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            train_df[feature_cols].values, train_y,
            eval_set=(valid_df[feature_cols].values, valid_y),
            early_stopping_rounds=50,
            verbose=100,
        )
        logger.info(f"CatBoost 训练完成: {self.model.best_iteration_} 轮")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        return self.model.predict(df[feature_cols].values)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(Path(path) / "model.cbm"))
        logger.info(f"CatBoost 已保存: {path}")

    def load(self, path: str) -> None:
        model_path = Path(path) / "model.cbm"
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))
        logger.info(f"CatBoost 已加载: {path}")
