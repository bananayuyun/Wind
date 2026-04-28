"""
流量模式生成器
支持：均匀随机、热点、转置、位翻转
"""
from __future__ import annotations

import random
from typing import List, Tuple

from config import TrafficConfig


class TrafficGenerator:
    def __init__(self, size: int = 4, seed: int = 42) -> None:
        self.size = size
        self.rng = random.Random(seed)
        self._hotspot_nodes: List[Tuple[int, int]] = []

    # ── 内部辅助 ───────────────────────────────────────────────────────────────

    def _rand_dst(self, exclude: Tuple[int, int]) -> Tuple[int, int]:
        while True:
            x = self.rng.randint(0, self.size - 1)
            y = self.rng.randint(0, self.size - 1)
            if (x, y) != exclude:
                return (x, y)

    def _ensure_hotspots(self) -> None:
        if not self._hotspot_nodes:
            n_hot = max(1, int(self.size * self.size * TrafficConfig.HOTSPOT_NODES_RATIO))
            all_nodes = [(x, y) for x in range(self.size) for y in range(self.size)]
            self._hotspot_nodes = self.rng.sample(all_nodes, n_hot)

    # ── 流量模式 ───────────────────────────────────────────────────────────────

    def uniform_random(self, injection_rate: float = 0.3) -> List[Tuple]:
        """均匀随机流量：每个节点以 injection_rate 的概率注入一个包"""
        packets = []
        for x in range(self.size):
            for y in range(self.size):
                if self.rng.random() < injection_rate:
                    packets.append(((x, y), self._rand_dst((x, y))))
        return packets

    def hotspot_traffic(
        self,
        injection_rate: float = 0.3,
        hotspot_ratio: float = None,
        hotspot_load: float = None,
    ) -> List[Tuple]:
        """
        热点流量：hotspot_load 比例的包打向少数热点节点，
        其余包均匀随机，形成明显的流量不均衡
        """
        hotspot_load = hotspot_load or TrafficConfig.HOTSPOT_TRAFFIC_RATIO
        self._ensure_hotspots()

        packets = []
        for x in range(self.size):
            for y in range(self.size):
                src = (x, y)
                if self.rng.random() < injection_rate:
                    valid_hot = [h for h in self._hotspot_nodes if h != src]
                    if valid_hot and self.rng.random() < hotspot_load:
                        dst = self.rng.choice(valid_hot)
                    else:
                        dst = self._rand_dst(src)
                    packets.append((src, dst))
        return packets

    def transpose_traffic(self, injection_rate: float = 0.3) -> List[Tuple]:
        """转置流量：src(x,y) → dst(y,x)，测试对角线通信"""
        packets = []
        for x in range(self.size):
            for y in range(self.size):
                if x != y and self.rng.random() < injection_rate:
                    packets.append(((x, y), (y, x)))
        return packets

    def bit_complement(self, injection_rate: float = 0.3) -> List[Tuple]:
        """位翻转流量：src(x,y) → dst(size-1-x, size-1-y)"""
        packets = []
        for x in range(self.size):
            for y in range(self.size):
                if self.rng.random() < injection_rate:
                    dst = (self.size - 1 - x, self.size - 1 - y)
                    if dst != (x, y):
                        packets.append(((x, y), dst))
        return packets

    def get_packets(self, pattern: str, injection_rate: float = 0.3) -> List[Tuple]:
        """按名称获取对应流量"""
        dispatch = {
            'uniform': self.uniform_random,
            'hotspot': self.hotspot_traffic,
            'transpose': self.transpose_traffic,
            'bit_complement': self.bit_complement,
        }
        return dispatch.get(pattern, self.uniform_random)(injection_rate)

    def reset_hotspots(self) -> None:
        """重置热点节点（新仿真前调用）"""
        self._hotspot_nodes = []
