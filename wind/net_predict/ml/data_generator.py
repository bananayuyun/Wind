"""
从 BookSim2 生成的 .npz 文件加载训练数据集。

数据由 booksim_data.py 脚本生成，保存在 data/booksim_{N}x{N}.npz。
若文件不存在，抛出 FileNotFoundError 并提示用户先运行生成脚本。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from config import MLConfig

_DATA_DIR = Path(__file__).parent.parent / "data"


def generate_dataset(
    grid_size: int = 4,
    seq_len: int = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载 BookSim 生成的训练数据集。

    Returns:
        X: shape (N, seq_len, n_nodes)  — 输入序列（缓冲利用率）
        y: shape (N, n_nodes)           — 热点标签（0/1）

    Raises:
        FileNotFoundError: 数据文件不存在时，提示运行 booksim_data.py
    """
    seq_len = seq_len or MLConfig.SEQ_LEN
    path = _DATA_DIR / f"booksim_{grid_size}x{grid_size}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"BookSim 数据文件不存在：{path}\n"
            f"请先运行：python3 booksim_data.py --grid {grid_size} --booksim <booksim_binary>"
        )

    d = np.load(str(path))
    X, y = d['X'], d['y']

    if X.shape[1] != seq_len:
        raise ValueError(
            f"数据 seq_len={X.shape[1]} 与配置 seq_len={seq_len} 不匹配，"
            f"请重新运行 booksim_data.py 生成数据。"
        )

    return X, y
