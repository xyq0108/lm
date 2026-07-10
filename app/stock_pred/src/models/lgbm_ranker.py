"""LGBM Ranker — 排序学习模型"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from src.models import BaseModel
from src.common.logger import get_logger

logger = get_logger(__name__)


class LGBMRankerModel(BaseModel):
    """LightGBM 排序模型 (LambdaRank)"""

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.params.setdefault("objective", "lambdarank")
        self.params.setdefault("boosting_type", "gbdt")
        self.params.setdefault("n_estimators", 500)
        self.params.setdefault("num_leaves", 31)
        self.params.setdefault("learning_rate", 0.05)
        self.params.setdefault("min_data_in_leaf", 50)
        self.params.setdefault("metric", "ndcg")
        self.params.setdefault("ndcg_eval_at", [5])
        self.model = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
            feature_cols: list) -> None:
        train_y = train_df["relevance"].values
        train_group = train_df.groupby("日期").size().values

        train_data = lgb.Dataset(train_df[feature_cols].values, label=train_y,
                                 group=train_group)

        if valid_df is not None:
            valid_y = valid_df["relevance"].values
            valid_group = valid_df.groupby("日期").size().values
            valid_data = lgb.Dataset(valid_df[feature_cols].values, label=valid_y,
                                     group=valid_group, reference=train_data)
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
            )
            logger.info(f"LGBM Ranker 训练完成: {self.model.best_iteration} 轮")
        else:
            # 全量训练：去掉早停参数，用满 n_estimators
            final_params = {k: v for k, v in self.params.items()
                            if k not in ("early_stopping_rounds", "verbose")}
            final_params.setdefault("n_estimators", 500)
            self.model = lgb.train(
                final_params,
                train_data,
            )
            logger.info("LGBM Ranker 全量训练完成（无验证集）")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        return self.model.predict(df[feature_cols].values)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(Path(path) / "model.joblib"))
        logger.info(f"LGBM Ranker 已保存: {path}")

    def load(self, path: str) -> None:
        model_path = Path(path) / "model.joblib"
        self.model = joblib.load(str(model_path))
        logger.info(f"LGBM Ranker 已加载: {path}")
