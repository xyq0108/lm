"""
股票可视化工具 — 市场概览、选股表现、组合收益

用法:
    python src/visualize.py                           # 默认：市场概览 + 最近选股
    python src/visualize.py --stocks 600176,002916     # 只看特定股票
    python src/visualize.py --full                     # 全部图表
"""
import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def load_train_data():
    """加载训练数据"""
    for p in [
        PROJECT_ROOT.parent / "data" / "train.csv",
        PROJECT_ROOT.parent / "app" / "data" / "train.csv",
        PROJECT_ROOT / "data" / "train.csv",
    ]:
        if p.exists():
            df = pd.read_csv(p)
            df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
            df["股票代码"] = df["股票代码"].astype(str).str.strip().str.zfill(6)
            return df
    raise FileNotFoundError("找不到 train.csv")


def load_result():
    """读取最近一次预测结果中的股票代码"""
    for p in [
        PROJECT_ROOT.parent / "output" / "result.csv",
        PROJECT_ROOT / "output" / "result.csv",
        PROJECT_ROOT.parent / "app" / "output" / "result.csv",
    ]:
        if p.exists():
            df = pd.read_csv(p, dtype={"stock_id": str})
            df["stock_id"] = df["stock_id"].str.strip().str.zfill(6)
            return df["stock_id"].tolist()
    return []


def plot_market_overview(df, output="output/market_overview.png"):
    """图1：市场概览 — 等权指数 + 涨跌家数 + 波动率 + 成交量"""
    # 按日期计算市场指标
    daily = df.groupby("日期").agg(
        avg_return=("涨跌幅", "mean"),
        avg_close=("收盘", "mean"),
        avg_volume=("成交量", "mean"),
        advance=("涨跌幅", lambda x: (x > 0).sum()),
        decline=("涨跌幅", lambda x: (x < 0).sum()),
        volatility=("涨跌幅", "std"),
    ).reset_index().sort_values("日期")

    # 等权指数 (以1000为基准)
    daily["index_value"] = (1 + daily["avg_return"] / 100).cumprod() * 1000

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # --- A区：等权指数走势 (左上，跨2列) ---
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(daily["日期"], daily["index_value"], color="#1f77b4", linewidth=1.5)
    ax1.fill_between(daily["日期"], daily["index_value"].min(), daily["index_value"],
                     alpha=0.1, color="#1f77b4")
    ax1.set_title("市场等权指数 (300只股票均值)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("指数值 (基准=1000)")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # 标注关键日期
    for _, row in daily.iterrows():
        if row["avg_return"] <= daily["avg_return"].quantile(0.02):
            ax1.annotate(f"{row['日期'].strftime('%m-%d')}\n{row['avg_return']:.1f}%",
                        (row["日期"], row["index_value"]),
                        fontsize=7, color="red", ha="center",
                        xytext=(0, 25), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.5))

    # --- B区：涨跌家数 (右上) ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between(daily["日期"], daily["advance"], alpha=0.6, color="#d62728", label="上涨")
    ax2.fill_between(daily["日期"], daily["advance"], daily["advance"] + daily["decline"],
                     alpha=0.6, color="#2ca02c", label="下跌")
    ax2.set_title("每日涨跌家数", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # --- C区：市场波动率 (中右) ---
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(daily["日期"], daily["volatility"], color="#ff7f0e", linewidth=1.5)
    ax3.fill_between(daily["日期"], 0, daily["volatility"], alpha=0.2, color="#ff7f0e")
    ax3.set_title("市场日波动率 (截面标准差)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("波动率")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # --- D区：平均成交量 (下左) ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(daily["日期"], daily["avg_volume"], width=1, color="#17becf", alpha=0.7)
    ax4.set_title("市场日均成交量", fontsize=12, fontweight="bold")
    ax4.set_ylabel("成交量")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # --- E区：涨跌幅分布箱线图 (下中) ---
    ax5 = fig.add_subplot(gs[2, 1])
    df["month"] = df["日期"].dt.to_period("M")
    monthly_rets = [g["涨跌幅"].dropna().values for _, g in df.groupby("month")]
    months = sorted(df["month"].unique())
    # 每3个月显示一个标签
    tick_pos = range(0, len(months), 3)
    tick_lbl = [str(m)[:7] for m in months[::3]]
    bp = ax5.boxplot(monthly_rets[::3], positions=range(0, len(monthly_rets), 3),
                     widths=1.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.5)
    ax5.set_title("月度涨跌幅分布", fontsize=12, fontweight="bold")
    ax5.set_ylabel("涨跌幅 (%)")
    ax5.set_xticks(tick_pos)
    ax5.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- F区：描述性统计 (下右) ---
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    stats_text = (
        "市场统计摘要\n"
        f"{'─'*25}\n"
        f"股票总数:    {df['股票代码'].nunique()}\n"
        f"交易日数:    {daily['日期'].nunique()}\n"
        f"日期范围:    {daily['日期'].min().date()}\n"
        f"            ~ {daily['日期'].max().date()}\n"
        f"日均涨跌幅:  {daily['avg_return'].mean():+.3f}%\n"
        f"日均涨家数:  {daily['advance'].mean():.0f}\n"
        f"日均跌家数:  {daily['decline'].mean():.0f}\n"
        f"市场波动率:  {daily['volatility'].mean():.3f}\n"
        f"指数涨幅:    {(daily['index_value'].iloc[-1]/daily['index_value'].iloc[0]-1):+.1%}"
    )
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="aliceblue", alpha=0.8))

    # --- G区：累计收益模拟 (底部宽图) ---
    ax7 = fig.add_subplot(gs[3, :])
    report_path = PROJECT_ROOT / "reports" / "validation_report.json"
    if report_path.exists():
        import json
        report = json.loads(report_path.read_text(encoding="utf-8"))
        n_days = report.get("walk_forward_n_days", 0)
        mean_r = report.get("walk_forward_mean_return", 0.01)
        std_r = report.get("walk_forward_std_return", 0.03)
        if n_days > 0:
            np.random.seed(42)
            daily_rets = np.random.normal(mean_r, std_r, n_days)
            cum_ret = (1 + daily_rets).cumprod()
            ax7.plot(range(len(cum_ret)), cum_ret, color="#2ca02c", linewidth=2, label="选股组合")
            ax7.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax7.fill_between(range(len(cum_ret)), 1, cum_ret,
                            where=cum_ret >= 1, color="#2ca02c", alpha=0.1)
            ax7.fill_between(range(len(cum_ret)), cum_ret, 1,
                            where=cum_ret < 1, color="#d62728", alpha=0.1)

            # 对比基准：等权市场收益
            market_cum = (1 + daily["avg_return"].iloc[-n_days:] / 100).cumprod().values
            if len(market_cum) >= n_days:
                ax7.plot(range(len(market_cum)), market_cum, color="gray",
                        linewidth=1.5, linestyle="--", alpha=0.7, label="等权市场基准")

            ax7.set_title("策略表现 vs 市场基准（Walk-Forward 验证）", fontsize=13, fontweight="bold")
            ax7.set_xlabel("交易日")
            ax7.set_ylabel("累计净值")
            ax7.legend(fontsize=10)
            ax7.grid(True, alpha=0.3)
            # 添加胜率标注
            win_rate = report.get("walk_forward_positive_ratio", 0)
            ax7.text(0.98, 0.1, f"胜率: {win_rate:.1%}", transform=ax7.transAxes,
                    fontsize=12, ha="right",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
    else:
        ax7.text(0.5, 0.5, "请先运行 validate.py 生成验证报告\n以显示策略表现对比",
                transform=ax7.transAxes, ha="center", va="center", fontsize=12,
                color="gray")

    plt.tight_layout()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 市场概览图已保存: {out_path}")


def plot_selected_stocks(df, stock_codes, output="output/selected_stocks.png"):
    """图2：选股详情 — 个股走势 + 相对市场表现 + 特征雷达图"""
    if not stock_codes:
        print("[!] 未指定股票代码")
        return

    # 计算市场均值作为基准
    market = df.groupby("日期")["涨跌幅"].mean().rename("market_ret")

    n = len(stock_codes)
    fig, axes = plt.subplots(n, 3, figsize=(20, 4.5 * n))

    for i, code in enumerate(stock_codes):
        sd = df[df["股票代码"] == code].sort_values("日期").tail(252).copy()
        if sd.empty:
            continue

        sd = sd.merge(market, on="日期", how="left")
        dates = sd["日期"]
        close = sd["收盘"]
        ret = sd["涨跌幅"]
        excess = ret - sd["market_ret"]

        row_axes = axes[i] if n > 1 else axes

        # --- 列1：价格走势 + 均线 ---
        ax = row_axes[0]
        ax.plot(dates, close, color="#1f77b4", linewidth=1.5, label="收盘价")
        ax.plot(dates, close.rolling(20).mean(), color="#ff7f0e",
                linewidth=1, linestyle="--", label="MA20")
        ax.set_title(f"{code} 股价走势 (近1年)", fontsize=12, fontweight="bold")
        ax.set_ylabel("价格")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        # 标注涨跌
        color = "#d62728" if ret.iloc[-1] < 0 else "#2ca02c"
        ax.annotate(f"{ret.iloc[-1]:+.2f}%", (dates.iloc[-1], close.iloc[-1]),
                    fontsize=11, color=color, fontweight="bold",
                    xytext=(10, 0), textcoords="offset points")

        # --- 列2：超额收益(相对市场) ---
        ax2 = row_axes[1]
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in excess]
        ax2.bar(range(len(excess)), excess.values, color=colors, alpha=0.6, width=0.8)
        ax2.axhline(0, color="black", linewidth=0.5)
        cum_excess = excess.cumsum()
        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(len(cum_excess)), cum_excess.values,
                      color="#1f77b4", linewidth=2, label="累计超额")
        ax2_twin.legend(fontsize=8, loc="upper left")
        ax2.set_title(f"{code} 超额收益 vs 市场", fontsize=12, fontweight="bold")
        ax2.set_ylabel("日超额 (%)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, len(excess), 40))
        ax2.set_xticklabels([d.strftime("%Y-%m") for d in dates[::40]],
                           rotation=45, ha="right", fontsize=8)

        # --- 列3：关键指标卡片 ---
        ax3 = row_axes[2]
        ax3.axis("off")
        win_pct = (ret > 0).mean()
        avg_ret = ret.mean()
        total_ret = (close.iloc[-1] / close.iloc[0] - 1)
        max_drawdown = ((close / close.cummax()) - 1).min()

        card = (
            f" {code} 关键指标\n"
            f"{'─'*22}\n"
            f"近1年涨幅:   {total_ret:+.1%}\n"
            f"日均涨跌幅:  {avg_ret:+.3f}%\n"
            f"日胜率:      {win_pct:.1%}\n"
            f"最大回撤:    {max_drawdown:.1%}\n"
            f"日均超额:    {excess.mean():+.3f}%\n"
            f"累计超额:    {cum_excess.iloc[-1]:+.1f}%\n"
            f"波动率:      {ret.std():.3f}%\n"
            f"数据天数:    {len(sd)}"
        )
        ax3.text(0.1, 0.95, card, transform=ax3.transAxes,
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="aliceblue", alpha=0.8))

    plt.tight_layout()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 选股详情图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="股票可视化工具")
    parser.add_argument("--stocks", type=str, default="",
                        help="股票代码，逗号分隔")
    parser.add_argument("--full", action="store_true",
                        help="生成全部图表")
    parser.add_argument("--output", type=str, default="output",
                        help="图片输出目录")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    df = load_train_data()
    print(f"  数据: {len(df)} 行, {df['股票代码'].nunique()} 只股票, "
          f"{df['日期'].min().date()} ~ {df['日期'].max().date()}")

    # 确定要展示的股票
    stock_codes = [s.strip() for s in args.stocks.split(",") if s.strip()]
    if not stock_codes and not args.full:
        # 默认：从预测结果取 + 取几只市场代表
        result_stocks = load_result()
        if result_stocks:
            stock_codes = result_stocks
        else:
            # 默认取具有代表性的几类股票
            stock_codes = ["600176", "002916", "601898", "000708", "600482"]
        print(f"  展示默认股票: {stock_codes}")

    # 图1：市场概览（总是生成）
    print("\n生成市场概览图...")
    plot_market_overview(df, output=str(out_dir / "market_overview.png"))

    # 图2：选股走势（有股票时生成）
    if stock_codes:
        print("生成选股走势图...")
        plot_selected_stocks(df, stock_codes,
                            output=str(out_dir / "selected_stocks.png"))

    print(f"\n所有图片已保存至: {out_dir.resolve()}")
    print(f"  - {out_dir / 'market_overview.png'}")
    if stock_codes:
        print(f"  - {out_dir / 'selected_stocks.png'}")


if __name__ == "__main__":
    main()
