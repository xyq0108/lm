"""
报告生成入口 — 生成实验报告和复现检查

用法:
    python ./stock_pred/src/make_report.py --config ./stock_pred/src/config/ensemble.yaml
"""
import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import read_yaml, save_json
from src.common.logger import setup_logger
from src.common.paths import ProjectPaths
from src.pipeline.export_pipeline import check_reproducibility
from src.features.feature_selection import get_feature_groups


def main():
    parser = argparse.ArgumentParser(description="2026 大数据挑战赛 — 报告生成")
    parser.add_argument("--config", type=str,
                        default=str(SRC_DIR / "config" / "ensemble.yaml"),
                        help="配置文件路径")
    parser.add_argument("--feature_cols", type=str, default=None,
                        help="特征列文件（用于特征分析）")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "reports"),
                        help="报告输出目录")
    args = parser.parse_args()

    config = read_yaml(args.config)
    logger = setup_logger("stock_pred")
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

    # 复现环境检查
    logger.info("--- 复现环境检查 ---")
    check_result = check_reproducibility(config)
    if check_result["all_pass"]:
        logger.info("所有检查通过 ✓")
    else:
        logger.warning("部分检查未通过:")
        for k, v in check_result["detail"].items():
            logger.warning(f"  - {k}: {v}")

    # 特征分析
    if args.feature_cols:
        try:
            from src.common.io import load_json
            cols = load_json(args.feature_cols)
            groups = get_feature_groups(cols)
            logger.info("--- 特征分析 ---")
            for group_name, group_cols in groups.items():
                if group_cols:
                    logger.info(f"  {group_name}: {len(group_cols)} 个特征")
        except Exception as e:
            logger.warning(f"特征分析失败: {e}")

    logger.info("报告生成完成")


if __name__ == "__main__":
    main()
