"""时间序列交叉验证划分（Walk-Forward）"""
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from src.common.logger import get_logger

logger = get_logger(__name__)


def build_walk_forward_folds(df: pd.DataFrame, fold_config: Union[int, List[dict]] = 5) -> List[Dict]:
    """
    构建 walk-forward 交叉验证划分。

    支持两种输入:
    1. fold_config 为 int — 自动滑动窗口划分
    2. fold_config 为 List[dict] — 使用显式日期范围，每项含:
        {"train_end": "YYYY-MM-DD", "valid_start": "YYYY-MM-DD", "valid_end": "YYYY-MM-DD"}

    返回:
        [{"train_idx": pd.Index, "valid_idx": pd.Index}, ...]
    """
    trading_days = sorted(pd.to_datetime(df["日期"].unique()))
    total_days = len(trading_days)

    # --- 模式1: 显式 fold 日期定义 ---
    if isinstance(fold_config, list):
        folds = []
        for i, spec in enumerate(fold_config):
            train_end = pd.Timestamp(spec["train_end"])
            valid_start = pd.Timestamp(spec["valid_start"])
            valid_end = pd.Timestamp(spec["valid_end"])

            train_days = [d for d in trading_days if d <= train_end]
            valid_days = [d for d in trading_days if valid_start <= d <= valid_end]

            if not train_days or not valid_days:
                logger.warning(f"Fold {i + 1}: 训练 {len(train_days)} 天, 验证 {len(valid_days)} 天, 跳过")
                continue

            train_idx = df[df["日期"].isin(train_days)].index
            valid_idx = df[df["日期"].isin(valid_days)].index

            folds.append({
                "train_idx": train_idx,
                "valid_idx": valid_idx,
            })

            logger.info(
                f"Fold {i + 1}/{len(fold_config)}: "
                f"训练 {len(train_days)} 天 ({train_days[0].strftime('%Y-%m-%d')} ~ "
                f"{train_days[-1].strftime('%Y-%m-%d')}), "
                f"验证 {len(valid_days)} 天 ({valid_days[0].strftime('%Y-%m-%d')} ~ "
                f"{valid_days[-1].strftime('%Y-%m-%d')})"
            )

        return folds

    # --- 模式2: 整数自动划分 ---
    n_folds = fold_config
    valid_size = total_days // (n_folds + 1)
    train_size = total_days - valid_size

    folds = []
    for i in range(n_folds):
        train_start = i * valid_size
        train_end = train_size + i * valid_size
        valid_end = train_end + valid_size

        if valid_end > total_days:
            logger.warning(f"Fold {i + 1}: 超出总交易日数，跳过")
            break

        train_days = trading_days[train_start:train_end]
        valid_days = trading_days[train_end:valid_end]

        train_idx = df[df["日期"].isin(train_days)].index
        valid_idx = df[df["日期"].isin(valid_days)].index

        folds.append({
            "train_idx": train_idx,
            "valid_idx": valid_idx,
        })

        logger.info(
            f"Fold {i + 1}/{n_folds}: "
            f"训练 {len(train_days)} 天 ({train_days[0].strftime('%Y-%m-%d')} ~ "
            f"{train_days[-1].strftime('%Y-%m-%d')}), "
            f"验证 {len(valid_days)} 天 ({valid_days[0].strftime('%Y-%m-%d')} ~ "
            f"{valid_days[-1].strftime('%Y-%m-%d')})"
        )

    return folds
