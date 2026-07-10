"""Tabular Dataset — 普通表格回归/分类 Dataset"""
import torch
from torch.utils.data import Dataset
import pandas as pd


class TabularDataset(Dataset):
    """普通表格数据集"""

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 label_col: str = "future_return_5d"):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        self.n_features = len(feature_cols)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
