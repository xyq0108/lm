"""Ranking Dataset — 排序学习用的 PyTorch Dataset"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class RankingDataset(Dataset):
    """LambdaRank 排序数据集，按日期分组为 queries"""

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 label_col: str = "relevance", group_col: str = "日期"):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        # query 分组边界
        self.groups = df[group_col].value_counts().sort_index().cumsum().tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_group_sizes(self):
        """返回每个 query 的文档数列表"""
        groups = [self.groups[0]] + [self.groups[i] - self.groups[i - 1]
                                      for i in range(1, len(self.groups))]
        return groups
