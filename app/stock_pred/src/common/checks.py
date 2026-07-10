"""数据校验和结果检查工具"""
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "股票代码", "日期", "开盘", "收盘", "最高", "最低",
    "成交量", "成交额", "换手率", "涨跌幅"
]

NUMERIC_COLUMNS = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率", "涨跌幅"]


def validate_schema(df: pd.DataFrame, strict: bool = True) -> list:
    """检查数据字段完整性，返回所有错误信息列表"""
    errors = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"缺少必要列: {col}")
    if errors:
        return errors

    try:
        pd.to_datetime(df["日期"])
    except Exception:
        errors.append("日期列无法解析为 datetime")

    stock_codes = df["股票代码"].astype(str).str.strip()
    if not stock_codes.str.match(r"^\d{6}$").all():
        errors.append("存在非 6 位数字的股票代码")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            non_num = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if non_num > 0:
                errors.append(f"{col} 存在 {non_num} 个非数值")

    if "开盘" in df.columns and (df["开盘"] <= 0).any():
        errors.append("存在开盘价 <= 0 的记录")

    if "成交量" in df.columns and (df["成交量"] < 0).any():
        errors.append("存在成交量 < 0 的记录")

    dup = df.duplicated(subset=["股票代码", "日期"]).sum()
    if dup > 0:
        errors.append(f"存在 {dup} 条重复的 股票代码+日期 记录")

    return errors


def validate_result_csv(result_df: pd.DataFrame) -> list:
    """检查 result.csv 是否符合提交规范，返回所有错误列表"""
    errors = []

    if result_df is None or len(result_df) == 0:
        errors.append("result_df 为空")
        return errors

    # 检查列名
    expected_cols = ["stock_id", "weight"]
    if list(result_df.columns) != expected_cols:
        errors.append(f"列名必须为 {expected_cols}，实际为 {list(result_df.columns)}")

    # 检查行数
    if len(result_df) > 5:
        errors.append(f"股票数量不能超过 5 只，当前 {len(result_df)} 只")
    if len(result_df) == 0:
        errors.append("股票数量不能为 0")

    # 检查股票代码
    ids = result_df["stock_id"].astype(str).str.strip()
    if not ids.str.match(r"^\d{6}$").all():
        errors.append("股票代码必须为 6 位数字")
    if ids.nunique() != len(result_df):
        errors.append("存在重复股票代码")

    # 检查权重
    weights = pd.to_numeric(result_df["weight"], errors="coerce")
    if weights.isna().any():
        errors.append("权重列存在非数值")
    if (weights <= 0).any():
        errors.append("所有权重必须 > 0")
    if weights.sum() > 1.0 + 1e-8:
        errors.append(f"权重和 {weights.sum():.4f} 超过 1.0")

    return errors


def check_future_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """检查是否因为未来数据泄漏而导致训练集中包含测试集日期"""
    train_dates = pd.to_datetime(train_df["日期"].unique()).max()
    test_dates = pd.to_datetime(test_df["日期"].unique()).min()
    if test_dates <= train_dates:
        return True
    return False
