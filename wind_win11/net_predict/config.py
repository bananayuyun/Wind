"""全局配置常量"""


class SimConfig:
    BUFFER_CAPACITY = 16        # 每个路由器缓冲区容量（packet 数）
    HOTSPOT_THRESHOLD = 0.8     # 热点判定阈值（缓冲利用率）
    DEFAULT_GRID = 4            # 默认 4×4 网格
    MAX_CYCLES = 2000           # 最大仿真周期（防止浏览器超时）
    MAX_HOPS = 32               # 最大跳数（防死锁）


class MLConfig:
    SEQ_LEN = 10                # 输入序列长度（过去 N 周期）
    PRED_HORIZON = 1            # 预测未来 M 周期
    HIDDEN_SIZE = 128           # LSTM 隐藏层维度
    NUM_LAYERS = 2              # LSTM 层数
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    DEFAULT_EPOCHS = 50
    MODEL_DIR = "models"


class TrafficConfig:
    DEFAULT_INJECTION_RATE = 0.3
    HOTSPOT_NODES_RATIO = 0.25    # 25% 节点作为热点目标
    HOTSPOT_TRAFFIC_RATIO = 0.65  # 65% 流量打向热点节点
