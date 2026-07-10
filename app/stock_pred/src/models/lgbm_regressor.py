"""LGBM Regressor — 回归预测模型"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from src.models import BaseModel
from src.common.logger import get_logger

logger = get_logger(__name__)


class LGBMRegressorModel(BaseModel):
    """LightGBM 回归模型"""

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.params.setdefault("objective", "regression")
        self.params.setdefault("boosting_type", "gbdt")
        self.params.setdefault("n_estimators", 500)
        self.params.setdefault("num_leaves", 31)
        self.params.setdefault("learning_rate", 0.05)
        self.params.setdefault("metric", "l2")
        self.model = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
            feature_cols: list) -> None:
        train_y = train_df["future_return_5d"].values
        train_data = lgb.Dataset(train_df[feature_cols].values, label=train_y)

        if valid_df is not None:
            valid_y = valid_df["future_return_5d"].values
            valid_data = lgb.Dataset(valid_df[feature_cols].values, label=valid_y,
                                     reference=train_data)
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
            )
            logger.info(f"LGBM Regressor 训练完成: {self.model.best_iteration} 轮")
        else:
            # 全量训练：去掉早停参数，用满 n_estimators
            final_params = {k: v for k, v in self.params.items()
                            if k not in ("early_stopping_rounds", "verbose")}
            final_params.setdefault("n_estimators", 500)
            self.model = lgb.train(
                final_params,
                train_data,
            )
            logger.info("LGBM Regressor 全量训练完成（无验证集）")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        return self.model.predict(df[feature_cols].values)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(Path(path) / "model.joblib"))
        logger.info(f"LGBM Regressor 已保存: {path}")

    def load(self, path: str) -> None:
        model_path = Path(path) / "model.joblib"
        self.model = joblib.load(str(model_path))
        logger.info(f"LGBM Regressor 已加载: {path}")
