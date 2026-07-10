"""数据加载模块"""
import pandas as pd
from pathlib import Path
from typing import Optional
from src.data.schema import validate_schema, standardize_columns
from src.common.logger import get_logger

logger = get_logger(__name__)


def load_train_data(path: str) -> pd.DataFrame:
    """读取训练数据，并做基础字段规范"""
    logger.info(f"读取训练数据: {path}")
    df = pd.read_csv(path)
    df = standardize_columns(df)

    # 解析日期并排序（处理带时间的日期格式）
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
    df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    # 股票代码统一为 6 位字符串（补零）
    df["股票代码"] = df["股票代码"].astype(str).str.strip().str.zfill(6)

    errors = validate_schema(df)
    if errors:
        for e in errors:
            logger.warning(f"Schema 检查: {e}")

    logger.info(f"数据读取完成: {len(df)} 行, {df['股票代码'].nunique()} 只股票, "
                f"日期范围 {df['日期'].min().date()} ~ {df['日期'].max().date()}")
    return df


def load_external_data(config: dict) -> dict:
    """读取外部公开数据。没有外部数据时返回空字典。"""
    external = {}
    data_dir = Path(config.get("paths", {}).get("data_dir", "./data")) / "external"
    if not data_dir.exists():
        logger.info("无外部数据目录")
        return external

    for fpath in data_dir.glob("*.csv"):
        name = fpath.stem
        try:
            df = pd.read_csv(fpath)
            external[name] = df
            logger.info(f"加载外部数据: {fpath.name} ({len(df)} 行)")
        except Exception as e:
            logger.warning(f"加载外部数据失败 {fpath.name}: {e}")

    if external:
        logger.info(f"共加载 {len(external)} 个外部数据集")
    return external
