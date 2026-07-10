"""验证流程 — Walk-Forward 验证和 Baseline 对比"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.common.logger import get_logger
from src.common.seed import set_seed
from src.common.io import save_json
from src.data.load_data import load_train_data
from src.data.clean_data import clean_ohlcv
from src.data.calendar import add_trading_index
from src.labels.make_label import make_future_return_label
from src.labels.make_relevance import add_all_relevance_labels
from src.features.feature_pipeline import FeaturePipeline
from src.validation.walk_forward import run_walk_forward, run_walk_forward_lgbm
from src.validation.metrics import topk_return, ndcg_at_k, hit_rate_at_k, baseline_win_rate
from src.validation.baseline_compare import compare_with_baseline
from src.common.checks import validate_schema

logger = get_logger(__name__)


def run_validation_pipeline(config: dict,
                            baseline_path: str = None) -> Dict:
    """
    验证流程 — 多维度评估模型性能。

    输出:
        - Walk-Forward 验证结果
        - TopK 收益统计
        - 与 Baseline 对比
    """
    set_seed(config["general"]["seed"])

    logger.info("=" * 50)
    logger.info("开始验证流程")

    # 准备数据
    df = load_train_data(config["paths"]["train_file"])
    errors = validate_schema(df)
    if errors:
        logger.warning(f"数据校验: {len(errors)} 个问题")

    df = clean_ohlcv(df)
    df = add_trading_index(df)
    df = make_future_return_label(df)
    df = add_all_relevance_labels(df)

    # 特征工程
    feat_cfg = config.get("features", {})
    feature_pipeline = FeaturePipeline(feat_cfg)
    df, feature_cols = feature_pipeline.fit_transform(df)

    # Walk-Forward 验证
    fold_config = config["validation"]["folds"]

    # 验证 LGBM Ranker
    logger.info("--- Walk-Forward: LGBM Ranker ---")
    lgb_ranker_cfg = {}
    for m in config.get("models", []):
        if m["type"] == "lgbm_ranker":
            lgb_ranker_cfg = m.get("params", {})
            break

    wf_result = run_walk_forward_lgbm(
        df, fold_config, feature_cols,
        label_col="future_return_5d",
        top_k=config["portfolio"]["top_k"],
        model_config=lgb_ranker_cfg
    )

    # 计算综合指标
    all_daily_returns = []
    for fr in wf_result["fold_results"]:
        if "daily_details" in fr:
            for dd in fr["daily_details"]:
                all_daily_returns.append(dd["topk_return"])

    returns_series = pd.Series(all_daily_returns)
    metrics = {
        "walk_forward_mean_return": float(returns_series.mean()),
        "walk_forward_std_return": float(returns_series.std()),
        "walk_forward_positive_ratio": float((returns_series > 0).mean()),
        "walk_forward_best_return": float(returns_series.max()),
        "walk_forward_worst_return": float(returns_series.min()),
        "walk_forward_n_days": len(returns_series),
    }

    # Baseline 对比
    if baseline_path:
        logger.info("--- Baseline 对比 ---")
        baseline_result = compare_with_baseline(returns_series, baseline_path=baseline_path)
        metrics["baseline_comparison"] = baseline_result
    else:
        logger.info("未指定 baseline 路径，跳过对比")

    # 保存验证报告
    report_path = Path(config["paths"]["report_dir"]) / "validation_report.json"
    save_json(metrics, str(report_path))
    logger.info(f"验证报告已保存: {report_path}")

    logger.info("=" * 50)
    logger.info("验证流程完成")

    return {
        "status": "success",
        "metrics": metrics,
        "n_features": len(feature_cols),
        "walk_forward_result": wf_result,
    }
