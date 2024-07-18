import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        num_channels: int,

        hid_dim: int = 128,
        nhead: int = 8,
        dropout: float = 0.2,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        # 1次元畳み込み層
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=hid_dim, kernel_size=2, stride=2, padding=1)

        # 位置エンコーダ層
        self.pos_encoder = PositionalEncoder(d_model=hid_dim, max_len=seq_len)
        
        # Transformerエンコーダ層
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        # 全結合層
        self.head = nn.Sequential(
            nn.Linear(in_features=hid_dim, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    # forwardメソッド
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = X.permute(2, 0, 1)
        X = self.pos_encoder(X)
        features = self.transformer_encoder(X)
        features = features.mean(dim=0)
        
        return self.head(features)


# 位置エンコーダ
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 位置エンコーディングの初期化
        pe = torch.zeros(max_len, d_model)

        # 位置インデックスの生成
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 周波数のスケールファクターの生成
        scale = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 偶数インデックスにはサイン関数
        pe[:, 0::2] = torch.sin(pos * scale)
        # 奇数インデックスにはコサイン関数
        pe[:, 1::2] = torch.cos(pos * scale)

        # バッジ処理に対応するために、次元を追加
        pe = pe.unsqueeze(1) # (max_len, 1, d_model)

        # バッファに登録
        self.register_buffer('pe', pe)

    # forwardメソッド
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

