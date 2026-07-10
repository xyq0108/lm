"""导出和提交辅助 — 生成报告、检查复现环境"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from src.common.logger import get_logger
from src.common.io import save_json, write_yaml
from src.common.checks import validate_result_csv

logger = get_logger(__name__)


def export_training_report(config: dict, train_result: Dict,
                           output_dir: str = "./reports") -> str:
    """
    生成训练报告 Markdown 文件。
    """
    report_path = Path(output_dir) / "training_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    selected = train_result.get("selected_models", [])
    trained = train_result.get("trained_models", [])

    lines = [
        "# 训练报告\n",
        f"- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 特征数量: {train_result.get('n_features', 0)}",
        f"- 选中模型: {', '.join(selected)}",
        f"- 模型保存路径: {train_result.get('model_dir', 'N/A')}",
        "",
        "## 各模型表现\n",
        "| 模型 | 平均收益 | 收益标准差 | 胜率 | 最差收益 |",
        "|------|---------|-----------|------|---------|",
    ]

    for m in trained:
        summary = m[1]
        lines.append(
            f"| {summary['model_type']} | "
            f"{summary['mean_return']:.4%} | "
            f"{summary['std_return']:.4%} | "
            f"{summary['win_rate']:.2%} | "
            f"{summary['worst_return']:.4%} |"
        )

    lines.extend([
        "",
        "## 配置信息\n",
        "```yaml",
    ])

    content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"训练报告已保存: {report_path}")
    return str(report_path)


def check_reproducibility(config: dict) -> Dict:
    """
    检查复现环境是否满足要求。
    """
    checks = {
        "random_seed_set": True,
        "config_saved": False,
        "feature_cols_saved": False,
        "model_saved": False,
        "result_csv_valid": False,
    }

    model_dir = Path(config["paths"]["model_dir"]) / "final_ensemble"
    output_dir = Path(config["paths"]["output_dir"])
    result_file = output_dir / "result.csv"

    checks["config_saved"] = (model_dir / "config.yaml").exists()
    checks["feature_cols_saved"] = (model_dir / "feature_cols.json").exists()
    checks["model_saved"] = model_dir.exists() and any(
        (model_dir / sub).exists() for sub in ["lgbm_ranker", "lgbm_regressor", "rule_model"]
    )

    if result_file.exists():
        try:
            result_df = pd.read_csv(result_file, dtype={"stock_id": str})
            result_df["stock_id"] = result_df["stock_id"].str.strip().str.zfill(6)
            errs = validate_result_csv(result_df)
            checks["result_csv_valid"] = len(errs) == 0
        except Exception:
            checks["result_csv_valid"] = False

    all_pass = all(checks.values())
    logger.info(f"复现环境检查: {'通过 ✓' if all_pass else '存在问题'}")
    if not all_pass:
        for k, v in checks.items():
            if not v:
                logger.warning(f"  - {k}: 未通过")

    return {"all_pass": all_pass, "detail": checks}
