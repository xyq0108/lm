"""
预测入口 — 一键生成最终提交文件

用法:
    python ./stock_pred/src/predict.py --config ./stock_pred/src/config/ensemble.yaml \\
        --model_dir ./model/final_ensemble --output ./output/result.csv
"""
import argparse
import sys
from pathlib import Path

# 当前文件路径：stock_pred/src/predict.py
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent          # stock_pred/src
PROJECT_ROOT = SRC_DIR.parent           # stock_pred

sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import read_yaml
from src.common.logger import setup_logger
from src.common.paths import ProjectPaths
from src.pipeline.predict_pipeline import run_predict_pipeline


def main():
    parser = argparse.ArgumentParser(description="2026 大数据挑战赛 — 预测入口")
    parser.add_argument("--config", type=str,
                        default=str(SRC_DIR / "config" / "ensemble.yaml"),
                        help="配置文件路径")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="最终模型目录（默认从配置文件路径自动推断）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认从配置文件路径自动推断）")
    parser.add_argument("--log_file", type=str, default=None,
                        help="日志文件路径")
    args = parser.parse_args()

    # 加载配置
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

    paths = ProjectPaths(
        data_dir=path_config.get("data_dir", str(project_root / "data")),
        model_dir=path_config.get("model_dir", str(project_root / "model")),
        output_dir=path_config.get("output_dir", str(project_root / "output")),
        temp_dir=path_config.get("temp_dir", str(project_root / "temp")),
        report_dir=path_config.get("report_dir", str(project_root / "reports")),
    )

    # 设默认值：model_dir = config路径 + final_ensemble, output = config输出路径 + result.csv
    model_dir = args.model_dir
    if model_dir is None:
        model_dir = str(Path(path_config["model_dir"]) / "final_ensemble")
    else:
        p = Path(model_dir)
        if not p.is_absolute():
            model_dir = str((project_root / p).resolve())

    output_path = args.output
    if output_path is None:
        output_path = str(Path(path_config["output_dir"]) / "result.csv")
    else:
        p = Path(output_path)
        if not p.is_absolute():
            output_path = str((project_root / p).resolve())

    # 运行预测流程
    result = run_predict_pipeline(config, model_dir, output_path)

    if result["status"] == "success":
        logger.info("预测成功完成！")
        logger.info(f"输出: {result['output_path']}")
        logger.info(f"选股: {result['selected_stocks']}")
        logger.info(f"权重: {[f'{w:.3f}' for w in result['weights']]}")
    else:
        logger.error(f"预测失败: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
