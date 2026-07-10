"""Sequence Dataset — 时序用的滑动窗口 Dataset（Transformer/RNN）"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.common.logger import get_logger

logger = get_logger(__name__)


class SequenceDataset(Dataset):
    """时序滑动窗口数据集 — 按股票分组构造序列样本"""

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 label_col: str = "future_return_5d",
                 seq_len: int = 20, stride: int = 1):
        self.seq_len = seq_len
        samples = []
        labels = []

        for code, group in df.groupby("股票代码"):
            group = group.sort_values("日期")
            values = group[feature_cols].values
            targets = group[label_col].values

            for i in range(0, len(values) - seq_len, stride):
                samples.append(values[i:i + seq_len])
                labels.append(targets[i + seq_len - 1])

        if not samples:
            logger.warning(f"序列数据集为空 (seq_len={seq_len})")
            self.samples = torch.empty(0, seq_len, len(feature_cols))
            self.labels = torch.empty(0)
        else:
            self.samples = torch.tensor(np.array(samples), dtype=torch.float32)
            self.labels = torch.tensor(np.array(labels), dtype=torch.float32)

        self.n_features = len(feature_cols)
        logger.info(f"序列数据集: {len(self)} 个样本, seq_len={seq_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
