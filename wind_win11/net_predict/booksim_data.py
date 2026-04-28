"""
BookSim2 数据生成脚本
用法：python3 booksim_data.py [--grid 4|8] [--booksim /path/to/booksim]

流程：
  1. 为各种流量模式/注入率生成 BookSim 配置文件
  2. 调用 BookSim 二进制，收集每周期缓冲区占用 CSV
  3. 解析 CSV，归一化为利用率
  4. 构造滑动窗口样本，保存为 .npz 供 ML 训练使用
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── 路径配置 ──────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
BOOKSIM_BIN = Path(__file__).parent.parent / "booksim2/src/booksim"
CFG_DIR     = PROJECT_DIR / "booksim_configs"
DATA_DIR    = PROJECT_DIR / "data"

# ── 参数 ──────────────────────────────────────────────────────────────────────

SEQ_LEN           = 10    # 输入序列长度
PRED_HORIZON      = 1     # 预测窗口
HOTSPOT_THRESHOLD = 0.8   # 热点判定阈值

# ── BookSim 流量模式定义 ───────────────────────────────────────────────────────

def traffic_configs(grid_size: int) -> List[dict]:
    """返回各流量模式 + 注入率的配置组合。
    需要高注入率（>0.5）才能使路由器缓冲区超过 80% 阈值。
    热点流量用 BookSim 括号格式：hotspot({node_ids},rate_ratio)
    """
    k = grid_size
    configs: List[dict] = []

    # ── 均匀随机流量：从低到高，在高注入率下产生全局拥塞 ──────────────────────
    for rate in [0.40, 0.55, 0.65, 0.75, 0.85]:
        configs.append({
            'name':           f'uniform_r{int(rate*100):02d}',
            'traffic':        'uniform',
            'injection_rate': rate,
        })

    # ── 热点流量：多组不同热点位置，注入率 ≥ 0.2 即可产生拥塞 ────────────────
    # 热点节点选取：中心区域附近的几对节点
    center = k // 2
    hotspot_groups = [
        [center * k + center,       center * k + center - 1],        # 主中心对
        [(center-1) * k + center,   (center-1) * k + center - 1],    # 中心上方
        [center * k + center,       (center-1) * k + center - 1],    # 对角
    ]
    # 过滤越界节点
    n_nodes = k * k
    hotspot_groups = [
        [n for n in grp if 0 <= n < n_nodes]
        for grp in hotspot_groups
    ]
    hotspot_groups = [grp for grp in hotspot_groups if len(grp) >= 1]

    for gi, grp in enumerate(hotspot_groups):
        hot_str = '{' + ','.join(str(n) for n in grp) + '}'
        for rate in [0.20, 0.30, 0.40, 0.50]:
            configs.append({
                'name':           f'hotspot_g{gi}_r{int(rate*100):02d}',
                'traffic':        f'hotspot({hot_str},4)',  # rate_ratio=4x
                'injection_rate': rate,
            })

    # ── 转置流量：对角线模式，中等注入率即可在特定路径产生拥塞 ──────────────
    for rate in [0.40, 0.55, 0.65]:
        configs.append({
            'name':           f'transpose_r{int(rate*100):02d}',
            'traffic':        'transpose',
            'injection_rate': rate,
        })

    # ── Tornado 流量 ────────────────────────────────────────────────────────
    for rate in [0.40, 0.55]:
        configs.append({
            'name':           f'tornado_r{int(rate*100):02d}',
            'traffic':        'tornado',
            'injection_rate': rate,
        })

    return configs


# ── 生成单个 BookSim 配置文件 ─────────────────────────────────────────────────

def write_cfg(
    grid_size: int,
    traffic: str,
    injection_rate: float,
    buf_stats_file: str,
    sample_period: int = 1000,
) -> str:
    """写临时配置文件，返回路径。
    使用 throughput 仿真模式，禁用 latency_thres，确保不会因拥塞提前中止。
    """
    base_cfg = CFG_DIR / f"base_mesh{grid_size}{grid_size}.cfg"
    # BookSim parser 不支持 include 指令，直接内联 base config 内容
    base_content = base_cfg.read_text()
    content = base_content + '\n'
    # 覆盖仿真控制参数
    content += f'traffic        = {traffic};\n'
    content += f'injection_rate = {injection_rate};\n'
    content += f'sample_period  = {sample_period};\n'
    content += f'max_samples    = 20;\n'       # 允许足够多的测量周期
    content += f'sim_type       = throughput;\n'  # 吞吐量模式：不需要延迟收敛
    content += f'buf_stats_file = {buf_stats_file};\n'

    fd, path = tempfile.mkstemp(suffix='.cfg', prefix='booksim_')
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


# ── 运行 BookSim ──────────────────────────────────────────────────────────────

def run_booksim(cfg_path: str, booksim_bin: Path, timeout: int = 300) -> bool:
    """运行 BookSim，返回是否可以继续处理 CSV。
    BookSim main() 约定：Run()=true → exit(-1) 即 returncode=255（收敛成功）
                        Run()=false → exit(0)  即 returncode=0 （不稳定）
    两种情况 CSV 都可能含有预热后数据，均视为成功，由调用方检查文件大小。
    """
    try:
        result = subprocess.run(
            [str(booksim_bin), cfg_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # 只有 ParseError 级别的错误（如语法错误）会产生其他退出码
        if result.returncode not in (0, 255):
            print(f"  [错误] BookSim 异常退出码 {result.returncode}")
            print(result.stderr[-500:] if result.stderr else '')
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [超时] BookSim 超过 {timeout}s")
        return False
    except FileNotFoundError:
        print(f"  [错误] 找不到 BookSim 二进制：{booksim_bin}")
        return False


# ── 解析 CSV 输出 ─────────────────────────────────────────────────────────────

def parse_buf_csv(csv_path: str, buf_cap: int) -> np.ndarray:
    """解析 BookSim 输出的缓冲区 CSV，归一化为利用率 [0,1]。

    Returns:
        shape (cycles, n_nodes)，dtype float32
    """
    rows = []
    with open(csv_path) as f:
        f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            # parts[0] = cycle，parts[1:] = per-router 缓冲占用（flits）
            occupancies = [int(x) for x in parts[1:]]
            rows.append(occupancies)

    if not rows:
        return np.zeros((0, 0), dtype=np.float32)

    arr = np.array(rows, dtype=np.float32)        # (cycles, n_routers)
    arr = np.clip(arr / max(buf_cap, 1), 0.0, 1.0)
    return arr


# ── 滑动窗口构造样本 ───────────────────────────────────────────────────────────

def make_samples(
    states: np.ndarray,
    seq_len: int = SEQ_LEN,
    pred_horizon: int = PRED_HORIZON,
    threshold: float = HOTSPOT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """从状态时序构造 (X, y) 样本对。

    Args:
        states: (T, N) 全网利用率时序
    Returns:
        X: (S, seq_len, N)
        y: (S, N)  — 未来 pred_horizon 步内任一步 >= threshold → 1
    """
    T, N = states.shape
    if T < seq_len + pred_horizon:
        return np.zeros((0, seq_len, N), np.float32), np.zeros((0, N), np.float32)

    X_list, y_list = [], []
    for i in range(T - seq_len - pred_horizon):
        X_win  = states[i: i + seq_len]
        future = states[i + seq_len: i + seq_len + pred_horizon]
        y_label = (future.max(axis=0) >= threshold).astype(np.float32)
        X_list.append(X_win)
        y_list.append(y_label)

    return np.array(X_list, np.float32), np.array(y_list, np.float32)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def generate(grid_size: int = 4, booksim_bin: Path = BOOKSIM_BIN) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    num_vcs     = 4
    vc_buf_size = 8
    # 2D Mesh 每路由器 inputs = 2*2+1 = 5（4方向 + 本地注入口）
    n_inputs = 2 * 2 + 1
    buf_cap  = n_inputs * num_vcs * vc_buf_size  # = 160 flits

    cfgs = traffic_configs(grid_size)

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    print(f"\n=== 生成 {grid_size}×{grid_size} Mesh 数据集（{len(cfgs)} 个仿真） ===\n")

    for i, cfg in enumerate(cfgs):
        print(f"[{i+1}/{len(cfgs)}] {cfg['name']}  inj={cfg['injection_rate']:.3f}", end='  ')

        fd, csv_path = tempfile.mkstemp(suffix='.csv', prefix='bufstats_')
        os.close(fd)

        cfg_path = write_cfg(
            grid_size      = grid_size,
            traffic        = cfg['traffic'],
            injection_rate = cfg['injection_rate'],
            buf_stats_file = csv_path,
            sample_period  = 1000,
        )

        ok = run_booksim(cfg_path, booksim_bin)
        os.remove(cfg_path)

        if not ok or not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
            print("跳过（仿真失败或无数据）")
            if os.path.exists(csv_path):
                os.remove(csv_path)
            continue

        states = parse_buf_csv(csv_path, buf_cap)
        os.remove(csv_path)

        if states.shape[0] < SEQ_LEN + PRED_HORIZON + 1:
            print(f"跳过（数据太少：{states.shape[0]} 行）")
            continue

        print(f"→ {states.shape[0]} 周期  max_util={states.max():.3f}", end='  ')

        X, y = make_samples(states)
        hotspot_ratio = y.mean()
        print(f"样本={len(X)}  热点率={hotspot_ratio:.3f}")

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        print("\n[错误] 没有生成任何数据，请检查 BookSim 路径和配置文件")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # 打乱
    idx = np.random.RandomState(42).permutation(len(X))
    X, y = X[idx], y[idx]

    out_path = DATA_DIR / f"booksim_{grid_size}x{grid_size}.npz"
    np.savez_compressed(
        str(out_path), X=X, y=y,
        grid_size=grid_size, seq_len=SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        threshold=HOTSPOT_THRESHOLD,
    )

    print(f"\n数据集已保存：{out_path}")
    print(f"  X shape: {X.shape}   y shape: {y.shape}")
    print(f"  热点样本比例: {y.mean():.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BookSim2 训练数据生成')
    parser.add_argument('--grid',    type=int, default=4,
                        choices=[4, 8], help='Mesh 规模（默认 4）')
    parser.add_argument('--booksim', type=str, default=str(BOOKSIM_BIN),
                        help='BookSim 二进制路径')
    args = parser.parse_args()

    generate(grid_size=args.grid, booksim_bin=Path(args.booksim))
