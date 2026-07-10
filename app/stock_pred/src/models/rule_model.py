"""Rule Model — 规则选股模型（基于因子打分的 Baseline）"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from src.models import BaseModel
from src.common.logger import get_logger

logger = get_logger(__name__)


class RuleModel(BaseModel):
    """规则模型 — 对因子加权打分，选综合分最高的股票"""

    def __init__(self, params: dict = None):
        self.params = params or {}
        # 因子及其权重: {"rsi_14": -1, "ret_20d": 1, "excess_ret": 1}
        self.factor_weights = self.params.get("factor_weights", {})
        self.factor_ranks = self.params.get("factor_ranks", [])

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
            feature_cols: list) -> None:
        """规则模型无需训练，使用配置的因子权重"""
        logger.info(f"规则模型: {len(self.factor_weights)} 个因子")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        scores = np.zeros(len(df))
        n_factors = 0

        # 对每个因子：横截面排名 [0,1] 后乘以权重
        for factor, weight in self.factor_weights.items():
            if factor not in df.columns:
                continue
            vals = df[factor].values
            # 转为排名打分
            ranks = pd.Series(vals).rank(pct=True, ascending=(weight > 0)).values
            scores += weight * ranks
            n_factors += 1

        if n_factors == 0:
            logger.warning("规则模型: 无可用因子，返回 0 分")
            return scores

        logger.info(f"规则模型预测完成: {n_factors} 个因子")
        return scores

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        logger.info(f"规则模型已保存: {path}")

    def load(self, path: str) -> None:
        model_path = Path(path) / "params.json"
        if model_path.exists():
            with open(model_path) as f:
                self.params = json.load(f)
            self.factor_weights = self.params.get("factor_weights", {})
        logger.info(f"规则模型已加载: {path}")
