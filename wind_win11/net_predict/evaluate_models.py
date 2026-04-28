"""
评估脚本 - 加载训练好的模型，计算论文表格所需的所有指标
用法：
  python3 evaluate_models.py
会自动扫描 models/ 目录下的 .pth 文件并逐个评估
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.model import build_model


def evaluate_model(model_path, data_dir="data"):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model_type = ckpt['model_type']
    grid_size = ckpt.get('grid_size', 4)
    n_nodes = ckpt.get('n_nodes', grid_size * grid_size)
    seq_len = ckpt.get('seq_len', 10)
    hidden_size = ckpt.get('hidden_size', 128)
    num_layers = ckpt.get('num_layers', 2)

    print(f"\n{'='*60}")
    print(f"模型: {os.path.basename(model_path)}")
    print(f"类型: {model_type} | 网格: {grid_size}x{grid_size} | 节点数: {n_nodes}")
    print(f"seq_len={seq_len}, hidden_size={hidden_size}, num_layers={num_layers}")
    print(f"训练时 best_val_loss={ckpt.get('best_val_loss', 'N/A')}")
    print(f"训练时 final_accuracy={ckpt.get('final_accuracy', 'N/A')}")

    model = build_model(model_type, n_nodes, seq_len, hidden_size, num_layers)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    data_path = os.path.join(data_dir, f"booksim_{grid_size}x{grid_size}.npz")
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return None

    d = np.load(data_path)
    X, y = d['X'], d['y']

    split = int(0.8 * len(X))
    X_val = X[split:]
    y_val = y[split:]

    print(f"验证集样本数: {len(X_val)}")
    print(f"验证集热点率: {y_val.mean():.4f} ({y_val.mean()*100:.2f}%)")

    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    tp, fp, fn, tn = 0, 0, 0, 0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            tp += ((preds == 1) & (yb == 1)).sum().item()
            fp += ((preds == 1) & (yb == 0)).sum().item()
            fn += ((preds == 0) & (yb == 1)).sum().item()
            tn += ((preds == 0) & (yb == 0)).sum().item()

            correct += (preds == yb).sum().item()
            total += yb.numel()

            all_probs.append(probs.numpy())
            all_labels.append(yb.numpy())

    acc = correct / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    fpr = fp / max(fp + tn, 1)
    tnr = tn / max(tn + fp, 1)

    print(f"\n--- 验证集评估结果 ---")
    print(f"总体准确率 (Accuracy):     {acc:.4f} ({acc*100:.2f}%)")
    print(f"热点精确率 (Precision):    {prec:.4f} ({prec*100:.2f}%)")
    print(f"热点召回率 (Recall):       {rec:.4f} ({rec*100:.2f}%)")
    print(f"热点F1分数 (F1-Score):     {f1:.4f} ({f1*100:.2f}%)")
    print(f"误报率 (FPR):             {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"真负率 (TNR/Specificity): {tnr:.4f} ({tnr*100:.2f}%)")
    print(f"\n混淆矩阵: TP={tp} FP={fp} FN={fn} TN={tn}")

    all_probs_np = np.concatenate(all_probs, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    dummy = torch.randn(1, seq_len, n_nodes)
    with torch.no_grad():
        times_list = []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = model(dummy)
            times_list.append(time.perf_counter() - t0)
    avg_ms = np.mean(times_list) * 1000
    median_ms = np.median(times_list) * 1000
    print(f"\nCPU推理延迟: 平均={avg_ms:.3f}ms, 中位数={median_ms:.3f}ms (200次)")

    results = {
        'model_path': model_path,
        'model_type': model_type,
        'grid_size': grid_size,
        'n_nodes': n_nodes,
        'total_params': total_params,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fpr': fpr,
        'tnr': tnr,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'val_hotspot_rate': float(y_val.mean()),
        'inference_avg_ms': avg_ms,
        'inference_median_ms': median_ms,
        'all_probs': all_probs_np,
        'all_labels': all_labels_np,
    }

    return results


def main():
    model_dir = "models"
    if not os.path.isdir(model_dir):
        print(f"找不到 {model_dir} 目录")
        return

    pth_files = sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith('.pth')
    ])

    if not pth_files:
        print("models/ 目录下没有 .pth 文件")
        return

    print(f"找到 {len(pth_files)} 个模型文件:")
    for f in pth_files:
        print(f"  - {os.path.basename(f)}")

    all_results = []
    for pth in pth_files:
        try:
            r = evaluate_model(pth)
            if r is not None:
                all_results.append(r)
        except Exception as e:
            print(f"评估 {pth} 失败: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print(f"\n\n{'='*60}")
        print("=" * 20 + "  论文表格数据汇总  " + "=" * 20)
        print(f"{'='*60}")

        for r in all_results:
            name = f"{r['model_type'].upper()} {r['grid_size']}x{r['grid_size']}"
            print(f"\n【{name}】")
            print(f"  参数量:        {r['total_params']:,}")
            print(f"  准确率:        {r['accuracy']*100:.2f}%")
            print(f"  精确率:        {r['precision']*100:.2f}%")
            print(f"  召回率:        {r['recall']*100:.2f}%")
            print(f"  F1:           {r['f1']*100:.2f}%")
            print(f"  误报率(FPR):   {r['fpr']*100:.2f}%")
            print(f"  推理延迟:      {r['inference_avg_ms']:.3f}ms")

        print(f"\n\n--- 表4-2: 预测精度对比 (4x4) ---")
        print(f"{'指标':<20} {'Attention-BiLSTM':<20} {'MLP':<20}")
        for metric, label in [
            ('accuracy', '准确率'),
            ('precision', '精确率'),
            ('recall', '召回率'),
            ('f1', 'F1分数'),
            ('fpr', '误报率'),
        ]:
            lstm4 = next((r for r in all_results if r['model_type']=='lstm' and r['grid_size']==4), None)
            mlp4 = next((r for r in all_results if r['model_type']=='mlp' and r['grid_size']==4), None)
            v1 = f"{lstm4[metric]*100:.2f}%" if lstm4 else "N/A"
            v2 = f"{mlp4[metric]*100:.2f}%" if mlp4 else "N/A"
            print(f"{label:<20} {v1:<20} {v2:<20}")

        print(f"\n--- 表4-3: 参数量与推理速度对比 (4x4) ---")
        lstm4 = next((r for r in all_results if r['model_type']=='lstm' and r['grid_size']==4), None)
        mlp4 = next((r for r in all_results if r['model_type']=='mlp' and r['grid_size']==4), None)
        if lstm4:
            print(f"Attention-BiLSTM: 参数量={lstm4['total_params']:,}, 推理延迟={lstm4['inference_avg_ms']:.3f}ms")
        if mlp4:
            print(f"MLP:              参数量={mlp4['total_params']:,}, 推理延迟={mlp4['inference_avg_ms']:.3f}ms")

        print(f"\n--- 表7-3: 8x8预测性能 ---")
        lstm8 = next((r for r in all_results if r['model_type']=='lstm' and r['grid_size']==8), None)
        mlp8 = next((r for r in all_results if r['model_type']=='mlp' and r['grid_size']==8), None)
        if lstm8:
            print(f"Attention-BiLSTM 8x8: acc={lstm8['accuracy']*100:.2f}% prec={lstm8['precision']*100:.2f}% rec={lstm8['recall']*100:.2f}% f1={lstm8['f1']*100:.2f}% fpr={lstm8['fpr']*100:.2f}%")
        if mlp8:
            print(f"MLP 8x8:              acc={mlp8['accuracy']*100:.2f}% prec={mlp8['precision']*100:.2f}% rec={mlp8['recall']*100:.2f}% f1={mlp8['f1']*100:.2f}% fpr={mlp8['fpr']*100:.2f}%")


if __name__ == '__main__':
    main()
