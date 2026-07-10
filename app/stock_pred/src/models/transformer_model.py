"""Transformer 时序模型 — 用 PyTorch 实现的简单 Attention 网络"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from src.models import BaseModel
from src.common.logger import get_logger

logger = get_logger(__name__)


class StockTransformer(nn.Module):
    """简单的 Transformer Encoder 用于股票时序"""

    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # 池化
        return self.output_head(x).squeeze(-1)


class TransformerStockModel(BaseModel):
    """基于 Transformer 的股票收益预测模型"""

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.seq_len = self.params.get("seq_len", 20)
        self.d_model = self.params.get("d_model", 64)
        self.nhead = self.params.get("nhead", 4)
        self.num_layers = self.params.get("num_layers", 2)
        self.epochs = self.params.get("epochs", 50)
        self.lr = self.params.get("lr", 1e-3)
        self.batch_size = self.params.get("batch_size", 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[StockTransformer] = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
            feature_cols: list) -> None:
        logger.info(f"Transformer 使用设备: {self.device}")
        self.model = StockTransformer(
            d_model=len(feature_cols), nhead=min(self.nhead, len(feature_cols)),
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train_df["future_return_5d"].values, dtype=torch.float32)
        X_valid = torch.tensor(valid_df[feature_cols].values, dtype=torch.float32)
        y_valid = torch.tensor(valid_df["future_return_5d"].values, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                # 添加 seq_len 维度 (batch, seq_len=1, features)
                loss = criterion(self.model(Xb.unsqueeze(1)), yb)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    train_pred = self.model(X_train.unsqueeze(1).to(self.device))
                    valid_pred = self.model(X_valid.unsqueeze(1).to(self.device))
                    train_loss = criterion(train_pred.cpu(), y_train)
                    valid_loss = criterion(valid_pred.cpu(), y_valid)
                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")

        logger.info("Transformer 训练完成")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        self.model.eval()
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X.unsqueeze(1)).cpu().numpy()
        return preds

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(Path(path) / "model.pt"))
        logger.info(f"Transformer 已保存: {path}")

    def load(self, path: str) -> None:
        model_path = Path(path) / "model.pt"
        self.model = StockTransformer(d_model=self.d_model, nhead=self.nhead,
                                      num_layers=self.num_layers).to(self.device)
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        logger.info(f"Transformer 已加载: {path}")
