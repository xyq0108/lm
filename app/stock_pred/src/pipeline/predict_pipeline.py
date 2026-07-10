"""预测主流程 — 生成最终提交文件 result.csv"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from src.common.logger import get_logger
from src.common.seed import set_seed
from src.common.io import load_json, write_csv
from src.common.checks import validate_schema, validate_result_csv
from src.data.load_data import load_train_data
from src.data.clean_data import clean_ohlcv
from src.data.calendar import add_trading_index
from src.features.feature_pipeline import FeaturePipeline
from src.models.lgbm_ranker import LGBMRankerModel
from src.models.lgbm_regressor import LGBMRegressorModel
from src.models.rule_model import RuleModel
from src.ensemble.rank_fusion import rank_average
from src.portfolio.select_topk import select_topk_stocks, build_latest_candidates
from src.portfolio.risk_filter import apply_risk_filter
from src.portfolio.weight_allocator import allocate_weights

logger = get_logger(__name__)


def load_model(model_dir: str, model_type: str):
    """加载指定类型的模型"""
    model_path = Path(model_dir) / model_type
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    if model_type == "lgbm_ranker":
        model = LGBMRankerModel()
    elif model_type == "lgbm_regressor":
        model = LGBMRegressorModel()
    elif model_type == "rule_model":
        model = RuleModel()
    elif model_type == "catboost":
        from src.models.catboost_model import CatBoostModel
        model = CatBoostModel()
    elif model_type == "transformer":
        from src.models.transformer_model import TransformerStockModel
        model = TransformerStockModel()
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    model.load(str(model_path))
    return model


def run_predict_pipeline(config: dict,
                         model_dir: str = None,
                         output_path: str = "./output/result.csv") -> Dict:
    """
    完整预测流程:
    1. 固定种子 → 2. 读取数据 → 3. 清洗 → 4. 特征工程
    5. 构建候选集 → 6. 加载模型 → 7. 预测 → 8. Rank融合
    9. 风险过滤 → 10. Top5选股 → 11. 权重分配 → 12. 校验输出
    """
    set_seed(config["general"]["seed"])

    if model_dir is None:
        model_dir = str(Path(config["paths"]["model_dir"]) / "final_ensemble")

    logger.info("=" * 50)
    logger.info("开始预测流程")

    # 2. 读取数据
    df = load_train_data(config["paths"]["train_file"])
    errors = validate_schema(df)
    if errors:
        logger.warning(f"数据校验发现 {len(errors)} 个问题")

    # 3. 清洗
    df = clean_ohlcv(df)

    # 4. 交易日历
    df = add_trading_index(df)

    # 5. 特征工程（使用训练时保存的配置）
    feature_pipeline = FeaturePipeline.load(model_dir)
    df, feature_cols = feature_pipeline.transform(df)

    # 6. 确定最新交易日
    latest_date = df["日期"].max()
    logger.info(f"最新交易日: {latest_date.date()}")

    # 7. 构造候选股票集
    candidate_df = build_latest_candidates(df, latest_date)

    if candidate_df.empty:
        logger.error("候选股票集为空！")
        return {"status": "error", "message": "无候选股票"}

    # 8. 加载融合配置
    ensemble_cfg = load_json(str(Path(model_dir) / "ensemble_config.json"))
    selected_models = ensemble_cfg.get("selected_models", [])
    weights_config = ensemble_cfg.get("ensemble_weights", {})

    if not selected_models:
        logger.warning("未找到融合配置，使用所有可用模型")
        selected_models = ["lgbm_ranker", "lgbm_regressor", "rule_model"]

    # 9. 加载并运行每个模型
    model_predictions = []
    model_names = []

    for model_type in selected_models:
        try:
            model = load_model(model_dir, model_type)
            logger.info(f"加载模型: {model_type}")

            pred_scores = model.predict(candidate_df, feature_cols)
            pred_df = candidate_df[["股票代码"]].copy()
            pred_df["日期"] = latest_date
            pred_df["score"] = pred_scores
            model_predictions.append(pred_df)
            model_names.append(model_type)
            logger.info(f"{model_type}: 完成预测")
        except Exception as e:
            logger.warning(f"模型 {model_type} 预测失败: {e}")
            continue

    if not model_predictions:
        logger.error("没有模型预测成功！")
        return {"status": "error", "message": "所有模型预测失败"}

    # 10. Rank 融合
    weights = [weights_config.get(name, 1.0) for name in model_names]
    # 归一化权重
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    fusion_df = rank_average(model_predictions, weights)
    logger.info(f"Rank融合完成: {len(model_names)} 个模型")

    # 11. 风险过滤
    fusion_df = apply_risk_filter(fusion_df, df)

    # 12. 选 Top5
    top_k = config.get("portfolio", {}).get("top_k", 5)
    selected = select_topk_stocks(fusion_df, top_k=top_k)

    if selected.empty:
        logger.error("未选出任何股票！")
        return {"status": "error", "message": "选股失败"}

    # 13. 权重分配
    result = allocate_weights(
        selected,
        method=config.get("portfolio", {}).get("weight_method", "equal"),
        max_single_weight=config.get("portfolio", {}).get("max_single_weight", 0.3),
    )

    # 14. 校验并输出
    errors = validate_result_csv(result)
    if errors:
        for e in errors:
            logger.error(f"输出校验失败: {e}")
        return {"status": "error", "message": errors}
    else:
        logger.info("输出校验通过 ✓")

    result.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"结果已保存: {output_path}")

    logger.info("=" * 50)
    logger.info("预测流程完成")

    return {
        "status": "success",
        "output_path": output_path,
        "n_stocks": len(result),
        "total_weight": float(result["weight"].sum()),
        "selected_stocks": result["stock_id"].tolist(),
        "weights": result["weight"].tolist(),
    }
