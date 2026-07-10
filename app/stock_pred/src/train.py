"""
训练入口 — 一键运行完整训练流程

推荐用法:
    python -m src.train

也可以指定配置文件:
    python -m src.train --config ./src/config/ensemble.yaml --mode final_train
"""

import argparse
import sys
from pathlib import Path

# 当前文件路径：stock_pred/src/train.py
CURRENT_FILE = Path(__file__).resolve()

# src 目录：stock_pred/src
SRC_DIR = CURRENT_FILE.parent

# 项目根目录：stock_pred
PROJECT_ROOT = SRC_DIR.parent

# 添加项目根目录到 sys.path
# 这样才能使用 from src.xxx import xxx
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import read_yaml
from src.common.logger import setup_logger
from src.common.paths import ProjectPaths
from src.pipeline.train_pipeline import run_train_pipeline
from src.pipeline.export_pipeline import export_training_report


def main():
    parser = argparse.ArgumentParser(description="2026 大数据挑战赛 — 训练入口")

    parser.add_argument(
        "--config",
        type=str,
        default=str(SRC_DIR / "config" / "ensemble.yaml"),
        help="配置文件路径"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="final_train",
        choices=["final_train", "debug"],
        help="训练模式"
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件路径"
    )

    args = parser.parse_args()

    # 加载配置
    config = read_yaml(args.config)

    logger = setup_logger(
        "stock_pred",
        log_file=args.log_file,
        level=config.get("general", {}).get("log_level", "INFO")
    )

    logger.info(f"加载配置文件: {args.config}")

    # 设置路径 — 将所有相对路径基于项目根目录解析为绝对路径
    path_config = config.get("paths", {})
    project_root = PROJECT_ROOT
    for key in ["data_dir", "train_file", "model_dir", "output_dir", "temp_dir", "report_dir"]:
        val = path_config.get(key)
        if val:
            p = Path(val)
            if not p.is_absolute():
                path_config[key] = str((project_root / p).resolve())

    paths = ProjectPaths(
        data_dir=path_config.get("data_dir", str(project_root / "data")),
        model_dir=path_config.get("model_dir", str(project_root / "model")),
        output_dir=path_config.get("output_dir", str(project_root / "output")),
        temp_dir=path_config.get("temp_dir", str(project_root / "temp")),
        report_dir=path_config.get("report_dir", str(project_root / "reports")),
    )

    # 运行训练流程
    result = run_train_pipeline(config, config_path=args.config)

    if result["status"] == "success":
        logger.info("训练成功完成！")

        # 生成报告
        report_path = export_training_report(config, result)
        logger.info(f"训练报告: {report_path}")
    else:
        logger.error(f"训练失败: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()