"""Backtest — 回测引擎（模拟每日调仓）"""
import pandas as pd
import numpy as np
from src.common.logger import get_logger

logger = get_logger(__name__)


def run_backtest(df: pd.DataFrame, score_col: str = "fusion_rank",
                 label_col: str = "future_return_5d",
                 top_k: int = 5, initial_capital: float = 1.0) -> dict:
    """简单回测：每日选 TopK 等权买入，持有到下一日卖出"""
    capital = initial_capital
    daily_values = [capital]
    trade_log = []

    for date in sorted(df["日期"].unique()):
        day_data = df[df["日期"] == date].copy()
        if len(day_data) < top_k:
            daily_values.append(capital)
            continue

        selected = day_data.nlargest(top_k, score_col)
        day_return = selected[label_col].mean()
        capital *= (1 + day_return)
        daily_values.append(capital)

        trade_log.append({
            "date": date,
            "n_stocks": len(selected),
            "day_return": day_return,
            "cumulative": capital,
        })

    total_return = capital / initial_capital - 1
    daily_returns = pd.Series([t["day_return"] for t in trade_log])
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)

    logger.info(f"回测完成: 总收益 {total_return:.4%}, Sharpe {sharpe:.3f}")
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": _max_drawdown(pd.Series(daily_values)),
        "trade_log": trade_log,
        "daily_values": daily_values,
    }


def _max_drawdown(values: pd.Series) -> float:
    peak = values.expanding().max()
    dd = (values - peak) / peak
    return float(dd.min())
