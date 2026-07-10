"""
验证入口 — Walk-Forward 验证和模式评估

用法:
    python ./stock_pred/src/validate.py --config ./stock_pred/src/config/default.yaml --baseline ./data/baseline_result.csv
"""
import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import read_yaml
from src.common.logger import setup_logger
from src.pipeline.valid_pipeline import run_validation_pipeline


def main():
    parser = argparse.ArgumentParser(description="2026 大数据挑战赛 — 验证入口")
    parser.add_argument("--config", type=str,
                        default=str(SRC_DIR / "config" / "ensemble.yaml"),
                        help="配置文件路径")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline 结果文件路径（可选）")
    parser.add_argument("--log_file", type=str, default=None,
                        help="日志文件路径")
    args = parser.parse_args()

    config = read_yaml(args.config)
    logger = setup_logger("stock_pred", log_file=args.log_file,
                          level=config.get("general", {}).get("log_level", "INFO"))
    logger.info(f"加载配置文件: {args.config}")

    # 解析配置中的相对路径为基于项目根目录的绝对路径
    path_config = config.get("paths", {})
    project_root = PROJECT_ROOT
    for key in ["data_dir", "train_file", "model_dir", "output_dir", "temp_dir", "report_dir"]:
        val = path_config.get(key)
        if val:
            p = Path(val)
            if not p.is_absolute():
                path_config[key] = str((project_root / p).resolve())

    result = run_validation_pipeline(config, args.baseline)

    if result["status"] == "success":
        metrics = result["metrics"]
        logger.info("=" * 50)
        logger.info("验证结果摘要:")
        logger.info(f"  Walk-Forward 平均收益: {metrics['walk_forward_mean_return']:.4%}")
        logger.info(f"  收益标准差: {metrics['walk_forward_std_return']:.4%}")
        logger.info(f"  胜率: {metrics['walk_forward_positive_ratio']:.2%}")
        logger.info(f"  特征数: {result.get('n_features', 0)}")

        if "baseline_comparison" in metrics:
            bc = metrics["baseline_comparison"]
            logger.info(f"  Baseline 胜率: {bc.get('胜率', 'N/A')}")
            logger.info("=" * 50)
    else:
        logger.error(f"验证失败: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
