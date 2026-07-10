"""Walk-Forward 验证 — 时间序列交叉验证"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


def run_walk_forward(df: pd.DataFrame, n_folds: int,
                     feature_cols: list, label_col: str = "future_return_5d",
                     top_k: int = 5) -> Dict:
    """通用 walk-forward 验证入口"""
    from src.data.split import build_walk_forward_folds
    folds = build_walk_forward_folds(df, n_folds)
    fold_results = []

    for i, fold in enumerate(folds):
        train_df = df.loc[fold["train_idx"]].copy()
        valid_df = df.loc[fold["valid_idx"]].copy()

        daily_details = []
        for date in sorted(valid_df["日期"].unique()):
            day_data = valid_df[valid_df["日期"] == date]
            if len(day_data) < top_k:
                continue
            # 按特征简单打分排序（可被具体模型替换）
            scores = day_data[feature_cols].mean(axis=1)
            top = day_data.iloc[np.argsort(-scores.values)][:top_k]
            day_ret = top[label_col].mean()
            daily_details.append({"date": date, "topk_return": day_ret})

        fold_results.append({"fold": i, "daily_details": daily_details})
        logger.info(f"Walk-Forward Fold {i+1}/{n_folds}: {len(daily_details)} 天")

    return {"fold_results": fold_results}


def run_walk_forward_lgbm(df: pd.DataFrame, n_folds: int,
                          feature_cols: list, label_col: str = "future_return_5d",
                          top_k: int = 5,
                          model_config: Optional[dict] = None) -> Dict:
    """使用 LGBM Ranker 做 walk-forward 验证"""
    from src.data.split import build_walk_forward_folds
    from src.models.lgbm_ranker import LGBMRankerModel

    folds = build_walk_forward_folds(df, n_folds)
    fold_results = []

    for i, fold in enumerate(folds):
        train_df = df.loc[fold["train_idx"]].copy()
        valid_df = df.loc[fold["valid_idx"]].copy()

        model = LGBMRankerModel(model_config or {})
        model.fit(train_df, valid_df, feature_cols)
        pred_scores = model.predict(valid_df, feature_cols)

        valid_df = valid_df.copy()
        valid_df["pred_score"] = pred_scores

        daily_details = []
        for date in sorted(valid_df["日期"].unique()):
            day_data = valid_df[valid_df["日期"] == date]
            if len(day_data) < top_k:
                continue
            top = day_data.nlargest(top_k, "pred_score")
            day_ret = top[label_col].mean()
            daily_details.append({"date": date, "topk_return": day_ret})

        fold_results.append({"fold": i, "daily_details": daily_details})
        logger.info(f"LGBM WF Fold {i+1}/{n_folds}: {len(daily_details)} 天")

    return {"fold_results": fold_results}
