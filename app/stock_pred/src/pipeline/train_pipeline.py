"""训练主流程 — 从数据到模型的完整训练管线"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.common.logger import get_logger, setup_logger
from src.common.seed import set_seed
from src.common.io import save_json, save_pickle, write_yaml, read_yaml
from src.common.paths import ProjectPaths
from src.common.checks import validate_schema
from src.data.load_data import load_train_data
from src.data.clean_data import clean_ohlcv
from src.data.calendar import add_trading_index
from src.labels.make_label import make_future_return_label
from src.labels.make_relevance import add_all_relevance_labels
from src.features.feature_pipeline import FeaturePipeline
from src.data.split import build_walk_forward_folds
from src.models.lgbm_ranker import LGBMRankerModel
from src.models.lgbm_regressor import LGBMRegressorModel
from src.models.rule_model import RuleModel
from src.validation.walk_forward import run_walk_forward
from src.ensemble.model_selection import select_models_for_ensemble
from src.portfolio.select_topk import select_topk_stocks
from src.portfolio.weight_allocator import allocate_weights

logger = get_logger(__name__)


def build_model(model_cfg: dict, main_config_path: str = None) -> object:
    """根据配置构建模型实例

    自动加载模型独立的配置文件（config_path），支持相对路径解析。
    """
    model_type = model_cfg["type"]

    # 加载模型独立的配置文件
    params = {}
    config_path = model_cfg.get("config_path")
    if config_path:
        try:
            p = Path(config_path)
            resolved = None
            if p.is_absolute():
                resolved = p
            else:
                # 候选解析路径（按优先级）
                candidates = []
                # ① 相对于主配置文件所在目录（model config 常在同一目录）
                if main_config_path:
                    candidates.append(Path(main_config_path).parent / config_path)
                    # ② 只用 basename（更常见的情况：同目录）
                    candidates.append(Path(main_config_path).parent / p.name)
                # ③ 相对当前工作目录
                candidates.append(Path(config_path).resolve())
                # ④ 相对于项目根目录（stock_pred/）
                if main_config_path:
                    project_root = Path(main_config_path).parent.parent
                    candidates.append(project_root / config_path)

                for cand in candidates:
                    cand = cand.resolve()
                    if cand.exists():
                        resolved = cand
                        break

            if resolved and resolved.exists():
                model_config = read_yaml(str(resolved))
                file_params = model_config.get("model", model_config)
                # 去掉元字段，保留模型参数
                for skip in ("type", "config_path", "active"):
                    file_params.pop(skip, None)
                params = file_params
            else:
                logger.warning(f"模型配置文件不存在（已尝试多种路径）: {config_path}")
        except Exception as e:
            logger.warning(f"加载模型配置文件失败 {config_path}: {e}")

    # 命令行/主配置中显式指定的 params 可覆盖文件中的值
    explicit_params = model_cfg.get("params", {})
    params.update(explicit_params)

    if model_type == "lgbm_ranker":
        return LGBMRankerModel(params)
    elif model_type == "lgbm_regressor":
        return LGBMRegressorModel(params)
    elif model_type == "rule_model":
        # 兼容 rule_weights → factor_weights 的命名
        if "rule_weights" in params and "factor_weights" not in params:
            params["factor_weights"] = params.pop("rule_weights")
        return RuleModel(params)
    elif model_type == "catboost":
        from src.models.catboost_model import CatBoostModel
        return CatBoostModel(params)
    elif model_type == "transformer":
        from src.models.transformer_model import TransformerStockModel
        return TransformerStockModel(params)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def train_and_validate_model(model, train_df: pd.DataFrame,
                              valid_df: pd.DataFrame,
                              feature_cols: list,
                              label_col: str = "future_return_5d",
                              top_k: int = 5,
                              model_type: str = "lgbm_ranker") -> Dict:
    """训练并验证单个模型"""
    # 训练
    model.fit(train_df, valid_df, feature_cols)

    # 验证期预测
    if model_type == "rule_model":
        pred_scores = model.predict(valid_df, feature_cols)
    else:
        pred_scores = model.predict(valid_df, feature_cols)

    pred_df = valid_df[["日期", "股票代码", label_col]].copy()
    pred_df["score"] = pred_scores

    # 模拟每日组合收益
    daily_returns = []
    for date in sorted(pred_df["日期"].unique()):
        day_data = pred_df[pred_df["日期"] == date].copy()
        if len(day_data) < top_k:
            continue
        day_topk = day_data.nlargest(top_k, "score")
        day_return = day_topk[label_col].mean()
        daily_returns.append({"date": date, "return": day_return})

    returns_series = pd.Series([r["return"] for r in daily_returns])
    report = {
        "mean_return": float(returns_series.mean()),
        "std_return": float(returns_series.std()),
        "min_return": float(returns_series.min()),
        "max_return": float(returns_series.max()),
        "positive_ratio": float((returns_series > 0).mean()),
        "n_valid_days": len(daily_returns),
        "daily_returns": returns_series,
    }
    return report


def run_train_pipeline(config: dict, config_path: str = None) -> Dict:
    """
    完整训练流程:
    1. 读取配置 → 2. 固定种子 → 3. 读取数据 → 4. Schema检查
    5. 数据清洗 → 6. 构造标签 → 7. 生成特征 → 8. 时间序列切分
    9. 训练多个模型 → 10. 验证 → 11. 选择融合模型 → 12. 保存
    """
    set_seed(config["general"]["seed"])
    save_dir = Path(config["paths"]["model_dir"]) / "final_ensemble"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. 读取数据
    logger.info("=" * 50)
    logger.info("开始训练流程")
    df = load_train_data(config["paths"]["train_file"])

    # 4. Schema检查
    errors = validate_schema(df)
    if errors:
        for e in errors:
            logger.warning(f"Schema: {e}")

    # 5. 数据清洗
    df = clean_ohlcv(df)

    # 6. 交易日历
    df = add_trading_index(df)

    # 7. 构造标签
    label_cfg = config.get("task", {})
    df = make_future_return_label(df,
                                   buy_shift=label_cfg.get("horizon_buy", 1),
                                   sell_shift=label_cfg.get("horizon_sell", 5))
    df = add_all_relevance_labels(df)

    # 8. 特征工程
    feat_cfg = config.get("features", {})
    feature_pipeline = FeaturePipeline(feat_cfg)
    df, feature_cols = feature_pipeline.fit_transform(df)
    logger.info(f"特征数量: {len(feature_cols)}")

    # 9. 时间序列划分
    folds = build_walk_forward_folds(df, config["validation"]["folds"])

    # 10. 训练和验证各模型
    model_configs = [m for m in config.get("models", []) if m.get("active", True)]
    trained_models = []

    for model_cfg in model_configs:
        model_type = model_cfg["type"]
        logger.info(f"--- 训练模型: {model_type} ---")

        # 为每个 fold 训练
        fold_reports = []
        for fold in folds:
            train_idx = fold["train_idx"]
            valid_idx = fold["valid_idx"]

            train_df = df.loc[train_idx].copy()
            valid_df = df.loc[valid_idx].copy()

            model = build_model(model_cfg, main_config_path=config_path)
            fold_report = train_and_validate_model(
                model, train_df, valid_df, feature_cols,
                top_k=config["portfolio"]["top_k"],
                model_type=model_type
            )
            fold_reports.append(fold_report)

        # 汇总
        all_returns = pd.concat([r["daily_returns"] for r in fold_reports])
        model_summary = {
            "model_type": model_type,
            "mean_return": float(all_returns.mean()),
            "std_return": float(all_returns.std()),
            "win_rate": float((all_returns > 0).mean()),
            "worst_return": float(all_returns.min()),
            "daily_returns": all_returns,
        }
        trained_models.append((model, model_summary))
        logger.info(f"{model_type}: 平均收益 {model_summary['mean_return']:.4%}")

    # 11. 选择融合模型
    model_results = [m[1] for m in trained_models]
    selected_indices = select_models_for_ensemble(model_results)
    selected_models = [trained_models[i] for i in selected_indices]
    logger.info(f"选中 {len(selected_models)}/{len(trained_models)} 个模型进入融合")

    # 12. 用全量数据重新训练选中的模型（所有 folds 数据）
    logger.info("使用全量数据重新训练最终模型...")
    final_models = []
    for orig_model, summary in selected_models:
        model_type = summary["model_type"]
        model_cfg = [m for m in model_configs if m["type"] == model_type][0]
        final_model = build_model(model_cfg, main_config_path=config_path)
        final_model.fit(df, None, feature_cols)
        final_models.append(final_model)

    # 13. 保存所有产物
    artifact_dir = save_dir
    feature_pipeline.save(str(artifact_dir))
    save_json(feature_cols, str(artifact_dir / "feature_cols.json"))
    write_yaml(config, str(artifact_dir / "config.yaml"))

    for model, (_, summary) in zip(final_models, selected_models):
        model_type = summary["model_type"]
        model.save(str(artifact_dir / model_type))

    # 保存融合配置
    ensemble_cfg = config.get("ensemble", {})
    save_json({
        "selected_models": [s[1]["model_type"] for s in selected_models],
        "ensemble_weights": ensemble_cfg.get("weights", {}),
        "ensemble_method": ensemble_cfg.get("method", "rank_average"),
    }, str(artifact_dir / "ensemble_config.json"))

    # 保存实验日志
    exp_log = pd.DataFrame([{
        "experiment_id": "final",
        "model_type": m[1]["model_type"],
        "feature_version": len(feature_cols),
        "mean_return": m[1]["mean_return"],
        "std_return": m[1]["std_return"],
        "win_rate": m[1]["win_rate"],
        "worst_return": m[1]["worst_return"],
    } for m in trained_models])
    exp_log.to_csv(str(Path(config["paths"]["report_dir"]) / "experiment_log.csv"),
                   index=False, encoding="utf-8")

    logger.info("=" * 50)
    logger.info("训练流程完成")
    logger.info(f"模型保存至: {artifact_dir}")

    return {
        "status": "success",
        "model_dir": str(artifact_dir),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "selected_models": [s[1]["model_type"] for s in selected_models],
        "final_models": final_models,
        "trained_models": trained_models,
    }
