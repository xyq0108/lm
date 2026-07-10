"""数据字段定义和 Schema 校验"""
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "股票代码", "日期", "开盘", "收盘", "最高", "最低",
    "成交量", "成交额", "换手率", "涨跌幅"
]

NUMERIC_COLUMNS = [
    "开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率", "涨跌幅"
]


def validate_schema(df: pd.DataFrame, strict: bool = True) -> list:
    """检查字段是否完整，数据类型是否合理。返回错误信息列表。"""
    errors = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"缺少必要列: {col}")

    if errors:
        return errors

    # 日期解析
    try:
        parsed = pd.to_datetime(df["日期"])
    except Exception:
        errors.append("日期列无法解析")
        return errors

    # 股票代码检查
    codes = df["股票代码"].astype(str).str.strip()
    bad_code = ~codes.str.match(r"^\d{6}$")
    if bad_code.any():
        n_bad = bad_code.sum()
        examples = codes[bad_code].unique()[:5]
        errors.append(f"存在 {n_bad} 个非 6 位股票代码，例如: {examples}")

    # 数值列检查
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            n_null = numeric.isna().sum()
            if n_null > 0:
                errors.append(f"{col} 存在 {n_null} 个无效数值")

    # 价格大于 0
    for price_col in ["开盘", "收盘", "最高", "最低"]:
        if price_col in df.columns:
            invalid = (df[price_col] <= 0).sum()
            if invalid > 0:
                errors.append(f"{price_col} 存在 {invalid} 个 <= 0 的值")

    # 成交量为非负
    if "成交量" in df.columns:
        neg_vol = (df["成交量"] < 0).sum()
        if neg_vol > 0:
            errors.append(f"成交量存在 {neg_vol} 个负值")

    # 重复检查
    dup = df.duplicated(subset=["股票代码", "日期"]).sum()
    if dup > 0:
        errors.append(f"存在 {dup} 条重复 (股票代码, 日期) 记录")

    return errors


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名，确保代码一致性"""
    df = df.copy()
    # 如果列名是英文，映射为中文
    col_map = {
        "code": "股票代码", "stock_code": "股票代码", "stock_id": "股票代码",
        "date": "日期", "trade_date": "日期",
        "open": "开盘", "close": "收盘", "high": "最高", "low": "最低",
        "volume": "成交量", "amount": "成交额",
        "turn": "换手率", "turnover": "换手率", "turnover_rate": "换手率",
        "pct_chg": "涨跌幅", "change": "涨跌幅", "pct_change": "涨跌幅",
    }
    df.rename(columns=col_map, inplace=True)
    return df
