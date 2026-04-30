# NoC 热点预测系统

基于深度学习的片上网络（Network-on-Chip, NoC）热点预测与智能路由优化系统。通过 Attention-BiLSTM 神经网络预测 Mesh NoC 中各路由器节点成为热点的概率，将预测结果融入路由决策，实现自适应路由，从而降低平均延迟、提升吞吐量。

## 功能特性

### 1. 网络仿真
- 支持 4×4、8×8 等多种 Mesh 网格拓扑
- 四种流量模式：均匀流量、热点流量、转置流量、位翻转流量
- 三种路由算法：XY 路由、奇偶转弯模型、ML 自适应路由
- Canvas 可视化网络状态，节点按缓冲利用率着色（绿→黄→红）
- ML 预测热点以蓝色光晕叠加显示
- 支持播放/暂停/拖动/重置控制

### 2. 模型训练
- 支持多种模型架构：
  - Attention-BiLSTM：双向 LSTM + 时间注意力机制（主模型）
  - MLP：多层感知机基线模型
  - LSTM：单向 LSTM（兼容旧版）
- SSE 实时推送训练进度（epoch、loss、accuracy）
- Chart.js 实时绘制 Loss 曲线和 Accuracy 曲线
- 自动保存验证集最佳模型 checkpoint
- Focal Loss 处理类别不平衡问题

### 3. 算法对比
- 并行运行 XY/奇偶转弯/ML 自适应三种路由算法
- 7 个性能指标对比：吞吐量、平均延迟、最大延迟、缓冲利用率、热点数、功耗、交付包数
- 汇总表格自动高亮最优/最差
- 5 个指标柱状图 + 吞吐量时序折线图

### 4. 热点预测
- 基于历史 10 个周期的全网状态预测未来 1 周期热点
- 支持实时预测和批量预测
- 预测结果可视化展示

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask 2.3+ |
| 深度学习 | PyTorch 2.0+ (CPU 版本) |
| 数值计算 | NumPy 1.24+ |
| 机器学习辅助 | scikit-learn 1.3+ |
| 前端框架 | 原生 JavaScript (ES6+)，Jinja2 模板 |
| 图表库 | Chart.js 4.4.0 (CDN) |
| 可视化 | HTML5 Canvas API |
| 实时通信 | Server-Sent Events (SSE) |
| 运行环境 | Python 3.12, WSL2 Linux |
| 数据生成 | 外部 BookSim2 仿真器（周期精确 NoC 仿真） |

## 安装指南

### 环境要求
- Python 3.12+
- pip 包管理器

### 安装步骤

1. 克隆项目
   ```bash
   git clone <repository-url>
   cd net_predict
   ```

2. 创建虚拟环境（推荐）
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. 安装依赖
   ```bash
   # 安装基础依赖
   pip install flask numpy scikit-learn

   # 安装 PyTorch CPU 版本（推荐，约200MB）
   pip install torch --index-url https://download.pytorch.org/whl/cpu

   # 或安装 GPU 版本（需要 CUDA）
   pip install torch
   ```

4. 启动应用
   ```bash
   python app.py
   ```

5. 访问应用
   
   打开浏览器访问：http://localhost:5000

   如果使用 WSL2，可能需要配置端口转发：
   ```bash
   netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=$(wsl hostname -I)
   ```

## 项目结构

```
net_predict/
├── app.py                      # Flask 主应用（Web 服务器 + REST API）
├── config.py                   # 全局配置常量（仿真/ML/流量参数）
├── booksim_data.py             # BookSim2 训练数据生成脚本
├── evaluate_models.py          # 模型评估脚本（输出论文表格数据）
├── create_docx.py              # Word 技术文档生成脚本
├── requirements.txt            # Python 依赖
├── install.txt                 # WSL 安装与启动指南
│
├── ml/                         # 机器学习模块
│   ├── __init__.py
│   ├── model.py                # 神经网络模型定义（Attention-BiLSTM, MLP）
│   ├── trainer.py              # 模型训练逻辑（支持 SSE 实时进度推送）
│   └── data_generator.py       # 从 .npz 加载训练数据集
│
├── simulator/                  # NoC 网络仿真器模块
│   ├── __init__.py
│   ├── mesh_network.py         # Mesh 网络核心（路由器、路由算法、仿真步进）
│   └── traffic.py              # 流量模式生成器（均匀/热点/转置/位翻转）
│
├── booksim_configs/            # BookSim2 仿真器配置文件
│   ├── base_mesh44.cfg         # 4x4 Mesh 基础配置
│   └── base_mesh88.cfg         # 8x8 Mesh 基础配置
│
├── models/                     # 训练好的模型权重（.pth 文件）
│   ├── lstm_4x4_*.pth
│   ├── lstm_8x8_*.pth
│   ├── mlp_4x4_*.pth
│   └── mlp_8x8_*.pth
│
├── data/                       # BookSim 生成的训练数据集
│   ├── booksim_4x4.npz
│   └── booksim_8x8.npz
│
├── templates/                  # Jinja2 HTML 模板
│   ├── base.html               # 基础布局（导航栏、Toast、全局 JS 工具）
│   ├── index.html              # 首页总览
│   ├── simulation.html         # 网络仿真页
│   ├── training.html           # 模型训练页
│   └── comparison.html         # 算法对比页
│
└── static/                     # 前端静态资源
    ├── css/main.css            # 全局样式（Apple/Linear 设计风格）
    └── js/
        ├── simulation.js       # Canvas 网络拓扑可视化引擎
        ├── training.js         # 训练页逻辑（SSE 订阅、Chart.js 曲线）
        └── comparison.js       # 对比页逻辑（柱状图、时序折线图）
```

## 配置说明

### 仿真配置 (SimConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BUFFER_CAPACITY | 16 | 每个路由器缓冲区容量（packet 数） |
| HOTSPOT_THRESHOLD | 0.8 | 热点判定阈值（缓冲利用率） |
| DEFAULT_GRID | 4 | 默认网格大小（4×4） |
| MAX_CYCLES | 2000 | 最大仿真周期 |
| MAX_HOPS | 32 | 最大跳数（防死锁） |

### 机器学习配置 (MLConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SEQ_LEN | 10 | 输入序列长度（过去 N 周期） |
| PRED_HORIZON | 1 | 预测未来 M 周期 |
| HIDDEN_SIZE | 128 | LSTM 隐藏层维度 |
| NUM_LAYERS | 2 | LSTM 层数 |
| DROPOUT | 0.2 | Dropout 比率 |
| LEARNING_RATE | 1e-3 | 学习率 |
| BATCH_SIZE | 64 | 批次大小 |
| DEFAULT_EPOCHS | 50 | 默认训练轮数 |

### 流量配置 (TrafficConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DEFAULT_INJECTION_RATE | 0.3 | 默认注入率 |
| HOTSPOT_NODES_RATIO | 0.25 | 热点节点比例（25%） |
| HOTSPOT_TRAFFIC_RATIO | 0.65 | 热点流量比例（65%） |

## API 接口

### 1. 仿真 API

POST /api/simulate

运行网络仿真，返回仿真结果和指标。

请求参数：
```json
{
  "grid_size": 4,
  "traffic_pattern": "hotspot",
  "num_cycles": 300,
  "routing_algorithm": "xy",
  "injection_rate": 0.3,
  "model_name": ""
}
```

响应：
```json
{
  "success": true,
  "metrics": {
    "throughput": 0.85,
    "avg_latency": 12.5,
    "avg_utilization": 0.45,
    "hotspot_count": 3,
    "power_mw": 150.2
  },
  "metrics_timeline": [...],
  "state_history": [...],
  "hotspot_predictions": [...],
  "packet_traces": [...],
  "hotspot_map": [...]
}
```

### 2. 训练 API

POST /api/train/start

启动模型训练任务。

请求参数：
```json
{
  "grid_size": 4,
  "model_type": "lstm",
  "seq_len": 10,
  "epochs": 50,
  "lr": 0.001,
  "batch_size": 64,
  "traffic_patterns": ["uniform", "hotspot", "transpose"],
  "n_cycles_per_pattern": 1000
}
```

响应：
```json
{
  "success": true,
  "session_id": "train_123456"
}
```

GET /api/train/stream/<session_id>

通过 SSE (Server-Sent Events) 获取训练进度。

事件格式：
```
data: {"type": "progress", "epoch": 1, "total_epochs": 50, "loss": 0.5, "val_loss": 0.48, "accuracy": 0.75}
```

### 3. 对比 API

POST /api/compare

对比多种路由算法的性能。

请求参数：
```json
{
  "grid_size": 4,
  "traffic_pattern": "hotspot",
  "num_cycles": 500,
  "injection_rate": 0.3,
  "model_name": "lstm_4x4_1234567890.pth"
}
```

响应：
```json
{
  "success": true,
  "algorithms": ["XY路由", "奇偶转弯", "ML自适应"],
  "metrics": {
    "avg_latency": [15.2, 12.8, 10.5],
    "throughput": [0.78, 0.82, 0.88],
    "avg_utilization": [0.42, 0.48, 0.52],
    "hotspot_count": [5, 3, 1],
    "power_mw": [180.5, 165.2, 152.8]
  },
  "timeline": {...},
  "details": {...},
  "ml_available": true
}
```

### 4. 模型管理 API

GET /api/models

获取所有已训练的模型列表。

响应：
```json
{
  "models": [
    {
      "name": "lstm_4x4_1234567890.pth",
      "grid_size": 4,
      "model_type": "lstm",
      "accuracy": 0.8523,
      "val_loss": 0.3245
    }
  ]
}
```

POST /api/predict

使用指定模型进行热点预测。

请求参数：
```json
{
  "model_name": "lstm_4x4_1234567890.pth",
  "state_sequence": [[0.1, 0.2, ...], ...]
}
```

响应：
```json
{
  "hotspot_probs": [0.85, 0.12, ...]
}
```

## 模型架构

### Attention-BiLSTM（主模型）

双向 LSTM + 时间注意力机制，具有以下特点：

1. 输入投影 + LayerNorm：去除绝对值偏置，学习相对变化趋势
2. 双向 LSTM：同时捕获"利用率在上升/下降"两种时序特征
3. 时间注意力：自适应加权每个时间步，聚焦最关键的历史周期
4. 残差连接：缓解梯度消失，加速收敛

模型结构：
```
Input (batch, seq_len, n_nodes)
    ↓
Input Projection (Linear + LayerNorm + ReLU + Dropout)
    ↓
Bidirectional LSTM (2 layers, hidden_size=128)
    ↓
Temporal Attention (Linear + Tanh + Linear + Softmax)
    ↓
Residual Connection + Fully Connected (Linear + LayerNorm + ReLU + Linear)
    ↓
Output (batch, n_nodes) - logits
```

### 损失函数

使用 Focal Loss 处理类别不平衡问题：
- α=0.25：降低易分类负样本权重
- γ=2.0：聚焦难区分的边界样本

### 优化器

- AdamW：带动量自适应学习率和权重衰减
- Cosine Annealing：余弦退火学习率调度

## 路由算法

### 1. XY 路由
确定性路由算法，先沿 X 方向移动，再沿 Y 方向移动。无死锁，但无法适应网络拥塞。

### 2. 奇偶转弯模型 (Odd-Even)
基于列号奇偶性禁止特定转弯方向，在候选方向中选择缓冲最轻的邻居。部分自适应，可避免死锁。

### 3. ML 自适应路由
综合评分 = 缓冲利用率 × 0.4 + 热点预测概率 × 0.6，选择评分最低的邻居。利用 ML 模型预测未来热点，实现智能路由。

## 使用流程

### 1. 网络仿真

1. 打开「网络仿真」页面
2. 选择流量模式（推荐「热点流量」）
3. 设置注入率（建议 0.2-0.4）和仿真周期（建议 300-500）
4. 选择路由算法
5. 点击「运行仿真」，观察 Canvas 实时着色和指标变化

### 2. 模型训练

1. 打开「模型训练」页面
2. 选择网格大小和模型类型
3. 设置训练参数（Epochs、学习率等）
4. 点击「开始训练」，观察 Loss 曲线实时更新
5. 训练完成后，模型自动保存到 models/ 目录

### 3. 算法对比

1. 打开「算法对比」页面
2. 选择已训练的 LSTM 模型
3. 设置流量模式和注入率
4. 点击「开始对比」，查看三种算法的指标对比图表

## 性能指标

系统提供以下性能指标对比：

| 指标 | 说明 |
|------|------|
| 吞吐量 (Throughput) | 单位时间内成功交付的包数 |
| 平均延迟 (Avg Latency) | 包从注入到交付的平均周期数 |
| 最大延迟 (Max Latency) | 所有已交付包中的最大延迟 |
| 缓冲利用率 (Avg Utilization) | 所有路由器缓冲区的平均占用率 |
| 热点数 (Hotspot Count) | 缓冲利用率超过阈值的路由器数量 |
| 功耗 (Power) | 基于 Orion 2.0 NoC 功耗模型估算 |
| 交付包数 (Delivered Packets) | 成功到达目的地的包总数 |

## 常见问题

### Q: ImportError: No module named 'torch'
A: 安装 PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q: address already in use
A: 杀死占用端口的进程：
```bash
lsof -ti:5000 | xargs kill -9
```

### Q: WSL 无法访问 localhost
A: 配置端口转发：
```bash
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=$(wsl hostname -I)
```

### Q: 训练速度慢
A: 
- 减少训练轮数（Epochs）
- 使用更小的网格大小（4×4）
- 减少每个模式的仿真周期数

### Q: 模型预测不准确
A:
- 增加训练数据量
- 延长训练轮数
- 调整学习率和批次大小
- 尝试不同的模型架构

### Q: 训练数据不存在
A: 运行 BookSim2 数据生成脚本：
```bash
python booksim_data.py
```

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
