"""
Walk-Forward 结果与 Baseline 对比分析
"""
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

# 切换到 app/ 目录
os.chdir(str(Path(__file__).parent / "app"))
sys.path.insert(0, str(Path.cwd() / "stock_pred"))

from src.common.logger import setup_logger
from src.common.seed import set_seed
from src.data.load_data import load_train_data
from src.data.clean_data import clean_ohlcv
from src.data.calendar import add_trading_index
from src.labels.make_label import make_future_return_label
from src.labels.make_relevance import add_all_relevance_labels
from src.features.feature_pipeline import FeaturePipeline
from src.validation.walk_forward import run_walk_forward_lgbm
from src.common.io import read_yaml

logger = setup_logger("compare", level="INFO")

# ========== 1. 加载配置和数据 ==========
config = read_yaml("stock_pred/src/config/validate_local.yaml")
set_seed(config["general"]["seed"])

print("正在加载数据...")
df = load_train_data(config["paths"]["train_file"])
df = clean_ohlcv(df)
df = add_trading_index(df)
df = make_future_return_label(df)
df = add_all_relevance_labels(df)

print("正在做特征工程...")
feat_cfg = config.get("features", {})
feature_pipeline = FeaturePipeline(feat_cfg)
df, feature_cols = feature_pipeline.fit_transform(df)

# ========== 2. Walk-Forward 验证 ==========
fold_config = config["validation"]["folds"]
lgb_ranker_cfg = {}
for m in config.get("models", []):
    if m["type"] == "lgbm_ranker":
        lgb_ranker_cfg = m.get("params", {})
        break

print("\n正在执行 Walk-Forward 验证...")
wf_result = run_walk_forward_lgbm(
    df, fold_config, feature_cols,
    label_col="future_return_5d",
    top_k=config["portfolio"]["top_k"],
    model_config=lgb_ranker_cfg
)

# ========== 3. 提取模型每日 Top5 收益和选股 ==========
model_daily = {}
for fr in wf_result["fold_results"]:
    if "daily_details" in fr:
        for dd in fr["daily_details"]:
            date_str = str(dd["date"])[:10]
            model_daily[date_str] = {
                "return": dd["topk_return"],
                "stocks": dd["topk_stocks"],
            }

model_returns = pd.Series({d: v["return"] for d, v in model_daily.items()}, name="model_return")
print(f"模型有效交易天数: {len(model_returns)}")

# ========== 4. 计算 Baseline 每日收益 ==========
# Baseline 选股
baseline_stocks = ["600023", "601668", "601018", "601818", "601186"]

trading_days = sorted(df["日期"].unique())
date_to_idx = {str(d)[:10]: i for i, d in enumerate(trading_days)}

# 构建股票价格索引
stock_data = {}
for sid in baseline_stocks:
    sdf = df[df["股票代码"] == sid].sort_values("日期")
    stock_data[sid] = sdf.set_index("日期")[["开盘", "future_return_5d"]]

def get_baseline_return(sid, current_date_str):
    """计算 baseline 股票在 date 的 T+1→T+5 收益"""
    if current_date_str not in date_to_idx:
        return None
    idx = date_to_idx[current_date_str]
    buy_offset, sell_offset = 1, 5
    if idx + sell_offset >= len(trading_days):
        return None
    t1_date = trading_days[idx + buy_offset]
    t5_date = trading_days[idx + sell_offset]

    sdf = stock_data.get(sid)
    if sdf is None:
        return None
    try:
        p1 = sdf.loc[t1_date, "开盘"]
        p5 = sdf.loc[t5_date, "开盘"]
        if p1 > 0:
            return p5 / p1 - 1.0
    except KeyError:
        pass
    return None

# 计算 Baseline 每日等权组合收益
baseline_daily = {}
for date in sorted(model_daily.keys()):
    if date not in date_to_idx:
        continue
    day_rets = [r for s in baseline_stocks
                if (r := get_baseline_return(s, date)) is not None]
    if day_rets:
        baseline_daily[date] = np.mean(day_rets)

baseline_returns = pd.Series(baseline_daily, name="baseline_return")

# ========== 5. 对比分析 ==========
common_dates = sorted(set(model_daily.keys()) & set(baseline_daily.keys()))
my_aligned = model_returns.loc[common_dates]
bl_aligned = baseline_returns.loc[common_dates]
excess = my_aligned - bl_aligned

wins = (my_aligned > bl_aligned).sum()
losses = (my_aligned < bl_aligned).sum()
ties = len(common_dates) - wins - losses

# 夏普比率
model_sharpe = my_aligned.mean() / my_aligned.std() * np.sqrt(252) if my_aligned.std() > 0 else 0
bl_sharpe = bl_aligned.mean() / bl_aligned.std() * np.sqrt(252) if bl_aligned.std() > 0 else 0

# 统计检验
try:
    from scipy.stats import binom_test
    p_value = binom_test(wins, n=wins+losses, p=0.5)
except ImportError:
    p_value = None

# ========== 输出报告 ==========
print("\n" + "=" * 68)
print("              Walk-Forward 验证 — 模型 vs Baseline 对比报告")
print("=" * 68)
print(f"\n模型: LGBM Ranker | 选股 Top-5 | 验证期: 2025-01-01 ~ 2025-09-30")
print(f"Baseline 选股: {', '.join(baseline_stocks)}")

print(f"\n{'─' * 68}")
print(f"{'指标':<22} {'模型':>12} {'Baseline':>12} {'超额':>12} {'优劣':>8}")
print(f"{'─' * 68}")
print(f"{'平均日收益':<22} {my_aligned.mean():>12.4%} {bl_aligned.mean():>12.4%} {excess.mean():>12.4%} {'YES' if excess.mean()>0 else 'NO':>8}")
print(f"{'收益标准差':<22} {my_aligned.std():>12.4%} {bl_aligned.std():>12.4%} {excess.std():>12.4%} {'':>8}")
print(f"{'最大日收益':<22} {my_aligned.max():>12.4%} {bl_aligned.max():>12.4%} {excess.max():>12.4%} {'':>8}")
print(f"{'最差日收益':<22} {my_aligned.min():>12.4%} {bl_aligned.min():>12.4%} {excess.min():>12.4%} {'':>8}")
print(f"{'中位数收益':<22} {my_aligned.median():>12.4%} {bl_aligned.median():>12.4%} {excess.median():>12.4%} {'':>8}")
print(f"{'正收益天数比':<22} {(my_aligned>0).mean():>12.2%} {(bl_aligned>0).mean():>12.2%} {'':>12} {'':>8}")
print(f"{'累计收益(求和)':<22} {my_aligned.sum():>12.4%} {bl_aligned.sum():>12.4%} {excess.sum():>12.4%} {'YES' if excess.sum()>0 else 'NO':>8}")
print(f"{'年化夏普比率':<22} {model_sharpe:>12.4f} {bl_sharpe:>12.4f} {model_sharpe-bl_sharpe:>+12.4f} {'YES' if model_sharpe>bl_sharpe else 'NO':>8}")
print(f"{'交易天数':<22} {len(my_aligned):>12} {len(bl_aligned):>12} {len(common_dates):>12} {'':>8}")
print(f"{'─' * 68}")

print(f"\n【胜率统计】")
print(f"  模型战胜 Baseline: {wins}/{len(common_dates)} ({wins/len(common_dates):.2%})")
print(f"  Baseline 战胜模型: {losses}/{len(common_dates)} ({losses/len(common_dates):.2%})")
print(f"  平局: {ties}")
if p_value is not None:
    sig = "★ 统计显著" if p_value < 0.05 else "不显著"
    print(f"  符号检验 p-value: {p_value:.4f} ({sig})")

print(f"\n【选股重叠分析】")
total_overlap = []
for date in common_dates:
    my_stocks = set(model_daily[date]["stocks"])
    bl_set = set(baseline_stocks)
    total_overlap.append(len(my_stocks & bl_set) / 5)
print(f"  模型与 Baseline 平均选股重叠率: {np.mean(total_overlap):.2%}")
print(f"  选股完全不同的天数: {sum(1 for o in total_overlap if o==0)}/{len(total_overlap)}")

print(f"\n【分时间段表现】")
for fold in fold_config:
    vs, ve = fold["valid_start"], fold["valid_end"]
    period_dates = [d for d in common_dates if vs <= d <= ve]
    if period_dates:
        p_my = my_aligned.loc[period_dates]
        p_bl = bl_aligned.loc[period_dates]
        p_ex = p_my - p_bl
        wins_p = (p_my > p_bl).sum()
        print(f"  {vs}~{ve}")
        print(f"    模型 {p_my.mean():>+.4%} | Baseline {p_bl.mean():>+.4%} | 超额 {p_ex.mean():>+.4%} | 胜率 {wins_p/len(period_dates):.0%} ({len(period_dates)}天)")

print(f"\n【累计收益曲线对比】")
print(f"  模型累计: {my_aligned.sum():>+.4%}")
print(f"  Baseline累计: {bl_aligned.sum():>+.4%}")
print(f"  超额累计: {excess.sum():>+.4%}")

# ========== 保存结果 ==========
comparison_df = pd.DataFrame({
    "日期": common_dates,
    "模型收益": my_aligned.values,
    "Baseline收益": bl_aligned.values,
    "超额收益": excess.values,
    "模型选股": [",".join(model_daily[d]["stocks"]) for d in common_dates],
})
comparison_df.to_csv("reports/baseline_comparison_detail.csv", index=False, encoding="utf-8-sig")
print(f"\n详细数据已保存: reports/baseline_comparison_detail.csv")

summary = {
    "验证期": {"开始": "2025-01-01", "结束": "2025-09-30"},
    "模型": {
        "平均日收益": round(my_aligned.mean(), 6),
        "标准差": round(my_aligned.std(), 6),
        "正收益比例": round((my_aligned > 0).mean(), 4),
        "夏普比率年化": round(model_sharpe, 4),
        "累计收益": round(my_aligned.sum(), 6),
    },
    "Baseline固定选股": {
        "股票列表": baseline_stocks,
        "平均日收益": round(bl_aligned.mean(), 6),
        "标准差": round(bl_aligned.std(), 6),
        "正收益比例": round((bl_aligned > 0).mean(), 4),
        "夏普比率年化": round(bl_sharpe, 4),
        "累计收益": round(bl_aligned.sum(), 6),
    },
    "对比": {
        "共同交易天数": len(common_dates),
        "战胜Baseline天数": int(wins),
        "战胜Baseline比例": round(wins/len(common_dates), 4),
        "平均超额收益": round(excess.mean(), 6),
        "统计显著性(p_value)": round(p_value, 4) if p_value else None,
        "选股重叠率": round(np.mean(total_overlap), 4),
    }
}
with open("reports/baseline_comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"汇总报告已保存: reports/baseline_comparison_summary.json")

print("\n" + "=" * 68)
print("分析完成！")
