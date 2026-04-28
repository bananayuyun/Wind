"""
模型训练逻辑，支持通过 queue.Queue 向 SSE 流推送实时进度
"""
from __future__ import annotations

import os
import queue
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import MLConfig, SimConfig
from ml.data_generator import generate_dataset
from ml.model import FocalLoss, build_model


def train_model(
    config: Dict,
    progress_queue: Optional[queue.Queue] = None,
) -> Dict:
    """
    在后台线程中训练热点预测模型，通过 progress_queue 推送进度事件。

    事件格式：
        {'type': 'status',   'message': str}
        {'type': 'progress', 'epoch': int, 'total_epochs': int,
         'loss': float, 'val_loss': float, 'accuracy': float}
        {'type': 'complete', 'model_name': str, 'final_accuracy': float}
        {'type': 'error',    'message': str}
    """

    def push(msg: Dict) -> None:
        if progress_queue is not None:
            progress_queue.put(msg)

    try:
        grid_size   = int(config.get('grid_size', 4))
        model_type  = config.get('model_type', 'lstm')
        seq_len     = int(config.get('seq_len', MLConfig.SEQ_LEN))
        pred_horizon = int(config.get('pred_horizon', MLConfig.PRED_HORIZON))
        epochs      = int(config.get('epochs', MLConfig.DEFAULT_EPOCHS))
        lr          = float(config.get('lr', MLConfig.LEARNING_RATE))
        batch_size  = int(config.get('batch_size', MLConfig.BATCH_SIZE))
        patterns    = config.get('traffic_patterns', ['uniform', 'hotspot', 'transpose'])
        n_cycles    = int(config.get('n_cycles_per_pattern', 1000))

        n_nodes = grid_size * grid_size

        # ── 生成数据 ──────────────────────────────────────────────────────────
        push({'type': 'status', 'message': '正在生成训练数据，请稍候...'})

        X, y = generate_dataset(
            grid_size=grid_size,
            seq_len=seq_len,
        )

        hotspot_ratio = float(y.mean())
        push({'type': 'status', 'message': (
            f'数据集就绪：{len(X)} 个样本，热点率={hotspot_ratio:.2%}，开始训练...'
        )})

        # ── 划分训练/验证集 ────────────────────────────────────────────────────
        split = int(0.8 * len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=batch_size, num_workers=0,
        )

        # ── 构建模型 ───────────────────────────────────────────────────────────
        torch.set_num_threads(4)
        model = build_model(model_type, n_nodes, seq_len)
        # AdamW 比 Adam 有更好的权重衰减（L2正则），减少过拟合
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # Focal Loss：比 pos_weight 更好处理不平衡，减少假正例
        # α=0.25 降低易分类负样本权重，γ=2 聚焦难区分边界样本
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1
        )

        best_val_loss = float('inf')
        best_state: Optional[Dict] = None

        # ── 训练循环 ───────────────────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            # Validate
            model.eval()
            val_loss = 0.0
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()
                    probs = torch.sigmoid(logits)
                    predicted = (probs > 0.5).float()
                    correct += (predicted == yb).sum().item()
                    total += yb.numel()
            val_loss /= max(len(val_loader), 1)
            accuracy = correct / total if total > 0 else 0.0

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            push({
                'type': 'progress',
                'epoch': epoch,
                'total_epochs': epochs,
                'loss': round(float(train_loss), 4),
                'val_loss': round(float(val_loss), 4),
                'accuracy': round(float(accuracy), 4),
            })

        # ── 保存最佳模型 ───────────────────────────────────────────────────────
        if best_state:
            model.load_state_dict(best_state)

        os.makedirs(MLConfig.MODEL_DIR, exist_ok=True)
        timestamp = int(time.time())
        model_name = f"{model_type}_{grid_size}x{grid_size}_{timestamp}.pth"
        model_path = os.path.join(MLConfig.MODEL_DIR, model_name)

        torch.save({
            'model_type': model_type,
            'grid_size': grid_size,
            'n_nodes': n_nodes,
            'seq_len': seq_len,
            'hidden_size': MLConfig.HIDDEN_SIZE,
            'num_layers': MLConfig.NUM_LAYERS,
            'state_dict': model.state_dict(),
            'best_val_loss': round(float(best_val_loss), 4),
            'final_accuracy': round(float(accuracy), 4),
        }, model_path)

        result = {
            'type': 'complete',
            'model_name': model_name,
            'final_accuracy': round(float(accuracy), 4),
            'best_val_loss': round(float(best_val_loss), 4),
        }
        push(result)
        return result

    except Exception as exc:
        err = {'type': 'error', 'message': str(exc)}
        push(err)
        raise
