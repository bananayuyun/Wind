"""
PyTorch 热点预测模型
- AttentionLSTMPredictor：双向LSTM + 时间注意力（主模型）
- MLPHotspotPredictor：展平 MLP（对比基线）
- LSTMHotspotPredictor：单向LSTM（旧版，保留兼容）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTMPredictor(nn.Module):
    """
    改进版：双向LSTM + 时间注意力机制

    改进点：
    1. 输入投影 + LayerNorm：去除绝对值偏置，学习相对变化趋势
    2. 双向 LSTM：同时捕获"利用率在上升/下降"两种时序特征
    3. 时间注意力：自适应加权每个时间步，聚焦最关键的历史周期
    4. 残差连接：缓解梯度消失，加速收敛

    输入: (batch, seq_len, n_nodes)
    输出: (batch, n_nodes) — logits，推理时需 torch.sigmoid()
    """

    def __init__(
        self,
        n_nodes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_size = hidden_size

        # 输入投影：将原始利用率映射到特征空间，LayerNorm 去偏置
        self.input_proj = nn.Sequential(
            nn.Linear(n_nodes, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # 双向 LSTM：每方向 hidden_size，输出 hidden_size*2
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_out_size = hidden_size * 2  # 双向

        # 时间注意力：为每个时间步打分，softmax 后加权求和
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, n_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_nodes)
        proj = self.input_proj(x)                       # (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(proj)                   # (batch, seq_len, hidden*2)

        # 时间注意力
        scores = self.attn(lstm_out)                    # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)          # 归一化到时间维
        context = (weights * lstm_out).sum(dim=1)       # (batch, hidden*2)

        return self.fc(context)                         # (batch, n_nodes)


class MLPHotspotPredictor(nn.Module):
    """
    MLP 基线模型（对比用，展平时间序列后输入）

    输入: (batch, seq_len, n_nodes) → reshape 为 (batch, seq_len*n_nodes)
    输出: (batch, n_nodes) — logits
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len

        self.net = nn.Sequential(
            nn.Linear(seq_len * n_nodes, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.size(0), -1))


class LSTMHotspotPredictor(nn.Module):
    """旧版单向 LSTM，保留用于加载历史模型权重"""

    def __init__(
        self,
        n_nodes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.lstm = nn.LSTM(
            input_size=n_nodes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class FocalLoss(nn.Module):
    """
    Focal Loss（用于高度不平衡分类）
    FL = -α(1-pt)^γ · log(pt)
    γ>0 降低易分类样本权重，让模型专注难区分的边界样本
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)                                   # 预测概率
        focal = self.alpha * (1 - pt) ** self.gamma * bce     # 难样本加权
        return focal.mean()


def build_model(
    model_type: str,
    n_nodes: int,
    seq_len: int = 10,
    hidden_size: int = 128,
    num_layers: int = 2,
) -> nn.Module:
    if model_type == 'lstm':
        return AttentionLSTMPredictor(n_nodes, hidden_size, num_layers)
    if model_type == 'mlp':
        return MLPHotspotPredictor(n_nodes, seq_len)
    raise ValueError(f"Unknown model_type: {model_type}")
