"""feature_pipeline — 特征工程管线，串联所有特征生成器"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from src.common.logger import get_logger
from src.features.technical_features import TechnicalFeatures
from src.features.cross_section_features import CrossSectionFeatures
from src.features.market_features import MarketFeatures
from src.features.alpha_features import AlphaFeatures
from src.features.feature_selection import FeatureSelector

logger = get_logger(__name__)


class FeaturePipeline:
    """特征工程管线：依次执行多个特征生成器，产出特征矩阵"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._generators = []
        self._build_generators()

    def _build_generators(self) -> None:
        active = self.config.get("active", [])
        if not active:
            active = ["technical", "cross_section", "market", "alpha"]

        registry = {
            "technical": TechnicalFeatures,
            "cross_section": CrossSectionFeatures,
            "market": MarketFeatures,
            "alpha": AlphaFeatures,
        }

        for name in active:
            if name in registry:
                sub_cfg = self.config.get(name, {})
                self._generators.append(registry[name](sub_cfg))

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """拟合并转换数据，返回 (df_with_features, feature_cols)"""
        df = df.copy()
        for gen in self._generators:
            df = gen.fit_transform(df)

        # 筛选非元数据的数值列作为特征
        feature_cols = self._select_feature_cols(df)
        logger.info(f"特征总数: {len(feature_cols)}")

        # 可选的特征选择
        selector_cfg = self.config.get("selection")
        if selector_cfg and selector_cfg.get("enabled", False):
            selector = FeatureSelector(selector_cfg)
            df, feature_cols = selector.select(df, feature_cols)
            logger.info(f"特征选择后: {len(feature_cols)} 个特征")

        return df, feature_cols

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """仅转换（推理阶段）"""
        df = df.copy()
        for gen in self._generators:
            df = gen.transform(df)
        feature_cols = self._select_feature_cols(df)
        return df, feature_cols

    def save(self, path: str) -> None:
        """保存管线配置"""
        save_path = Path(path) / "feature_pipeline_config.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        logger.info(f"特征管线配置已保存: {save_path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """从保存的配置恢复管线"""
        load_path = Path(path) / "feature_pipeline_config.json"
        if not load_path.exists():
            logger.warning(f"特征管线配置不存在: {load_path}，使用默认配置")
            return cls({})
        with open(load_path) as f:
            config = json.load(f)
        return cls(config)

    def _select_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """识别数值特征列"""
        exclude = {"股票代码", "日期", "trading_idx",
                   "future_return_1d", "future_return_5d",
                   "future_open_1d", "relevance", "label"}
        return [c for c in df.columns
                if c not in exclude
                and df[c].dtype in ("float64", "float32", "int64", "int32")]
