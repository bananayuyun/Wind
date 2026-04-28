"""
NoC Mesh 网络仿真器
支持 4×4 / 8×8，三种路由算法：XY路由、奇偶转弯模型、ML自适应路由
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SimConfig


# ──────────────────────────────────────────────────────────────────────────────
# 数据包
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Packet:
    src: Tuple[int, int]
    dst: Tuple[int, int]
    inject_cycle: int
    current_pos: Tuple[int, int] = field(default=None)
    hops: int = 0
    arrival_cycle: int = -1
    stall_cycles: int = 0
    path: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.current_pos is None:
            self.current_pos = self.src
        if not self.path:
            self.path.append(self.src)

    @property
    def is_delivered(self) -> bool:
        return self.arrival_cycle >= 0

    @property
    def latency(self) -> int:
        return self.arrival_cycle - self.inject_cycle if self.is_delivered else -1


# ──────────────────────────────────────────────────────────────────────────────
# 路由器节点
# ──────────────────────────────────────────────────────────────────────────────

class Router:
    def __init__(self, x: int, y: int, buffer_capacity: int = 16) -> None:
        self.x = x
        self.y = y
        self.buffer_capacity = buffer_capacity
        self.buffer: deque = deque()
        self.packets_forwarded: int = 0
        self.packets_received: int = 0

    @property
    def buffer_depth(self) -> int:
        return len(self.buffer)

    @property
    def utilization(self) -> float:
        return min(self.buffer_depth / self.buffer_capacity, 1.0)

    def can_accept(self) -> bool:
        return self.buffer_depth < self.buffer_capacity


# ──────────────────────────────────────────────────────────────────────────────
# Mesh 网络
# ──────────────────────────────────────────────────────────────────────────────

class MeshNetwork:
    """
    规格：size × size Mesh，支持 4×4 / 8×8。
    坐标：x=列（0=左），y=行（0=上），节点索引 = x * size + y
    """

    def __init__(self, size: int = 4, buffer_capacity: int = None) -> None:
        self.size = size
        self.buffer_capacity = buffer_capacity or SimConfig.BUFFER_CAPACITY
        self.max_hops = max(SimConfig.MAX_HOPS, size * 4)
        self.cycle = 0

        # 网格，routers[x][y]
        self.routers: List[List[Router]] = [
            [Router(x, y, self.buffer_capacity) for y in range(size)]
            for x in range(size)
        ]

        self.delivered: List[Packet] = []
        self.dropped: int = 0
        self._flit_hops: int = 0   # 累计 flit 跳数，用于功耗估算

        self.state_history: List[np.ndarray] = []
        self.link_usage: Dict[Tuple, int] = {}

    # ── 拓扑辅助 ──────────────────────────────────────────────────────────────

    def get_router(self, pos: Tuple[int, int]) -> Router:
        return self.routers[pos[0]][pos[1]]

    def get_neighbors(self, pos: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
        x, y = pos
        nb: Dict[str, Tuple[int, int]] = {}
        if x > 0:             nb['W'] = (x - 1, y)
        if x < self.size - 1: nb['E'] = (x + 1, y)
        if y > 0:             nb['N'] = (x, y - 1)
        if y < self.size - 1: nb['S'] = (x, y + 1)
        return nb

    # ── 路由算法 ───────────────────────────────────────────────────────────────

    def xy_next_hop(
        self,
        current: Tuple[int, int],
        dst: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """XY 路由：先水平（X方向）后垂直（Y方向），确定性，无死锁"""
        cx, cy = current
        dx, dy = dst
        if cx < dx: return (cx + 1, cy)   # East
        if cx > dx: return (cx - 1, cy)   # West
        if cy < dy: return (cx, cy + 1)   # South
        if cy > dy: return (cx, cy - 1)   # North
        return None  # 已到达

    def odd_even_next_hop(
        self,
        current: Tuple[int, int],
        dst: Tuple[int, int],
        came_from: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        奇偶转弯模型（Odd-Even Turn Model）：
        - 偶列（x%2==0）：禁止 N/S → E 方向转弯（从竖转横时只能向左/West）
        - 奇列（x%2==1）：禁止 E/W → N/S 之后再向 West 转弯
        自适应选择缓冲最轻的方向，无死锁
        """
        cx, cy = current
        dx, dy = dst
        ex, ey = dx - cx, dy - cy

        if ex == 0 and ey == 0:
            return None

        neighbors = self.get_neighbors(current)
        # 最短路径候选
        candidates: List[Tuple[int, int]] = []
        if ex > 0 and 'E' in neighbors: candidates.append(neighbors['E'])
        elif ex < 0 and 'W' in neighbors: candidates.append(neighbors['W'])
        if ey > 0 and 'S' in neighbors: candidates.append(neighbors['S'])
        elif ey < 0 and 'N' in neighbors: candidates.append(neighbors['N'])

        if not candidates:
            return self.xy_next_hop(current, dst)

        # 奇偶约束过滤
        def _dir(a: Tuple[int, int], b: Tuple[int, int]) -> str:
            if b[0] > a[0]: return 'E'
            if b[0] < a[0]: return 'W'
            if b[1] > a[1]: return 'S'
            return 'N'

        in_dir = _dir(came_from, current) if came_from else None

        def is_allowed(nxt: Tuple[int, int]) -> bool:
            out_dir = _dir(current, nxt)
            if cx % 2 == 0:
                # 偶列：从垂直方向来，不能向 E 走（允许向 W）
                if in_dir in ('N', 'S') and out_dir == 'E':
                    return False
            else:
                # 奇列：从水平方向来，不能向 W 走（允许向 E）
                if in_dir in ('E', 'W') and out_dir == 'W':
                    return False
            return True

        allowed = [c for c in candidates if is_allowed(c)]
        if not allowed:
            # 降级到 XY 路由
            return self.xy_next_hop(current, dst)

        # 选缓冲最轻的邻居（负载均衡）
        return min(allowed, key=lambda p: self.get_router(p).utilization)

    def ml_adaptive_next_hop(
        self,
        current: Tuple[int, int],
        dst: Tuple[int, int],
        hotspot_probs: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        基于 ML 预测的自适应路由：
        综合评分 = 当前缓冲利用率×0.4 + 热点预测概率×0.6，选分数最低的邻居
        """
        if hotspot_probs is None:
            return self.xy_next_hop(current, dst)

        cx, cy = current
        dx, dy = dst
        ex, ey = dx - cx, dy - cy

        if ex == 0 and ey == 0:
            return None

        neighbors = self.get_neighbors(current)
        candidates: List[Tuple[int, int]] = []
        if ex > 0 and 'E' in neighbors: candidates.append(neighbors['E'])
        elif ex < 0 and 'W' in neighbors: candidates.append(neighbors['W'])
        if ey > 0 and 'S' in neighbors: candidates.append(neighbors['S'])
        elif ey < 0 and 'N' in neighbors: candidates.append(neighbors['N'])

        if not candidates:
            return self.xy_next_hop(current, dst)

        def score(pos: Tuple[int, int]) -> float:
            node_idx = pos[0] * self.size + pos[1]
            buf_util = self.get_router(pos).utilization
            pred = float(hotspot_probs[node_idx]) if node_idx < len(hotspot_probs) else 0.0
            return buf_util * 0.4 + pred * 0.6

        return min(candidates, key=score)

    # ── 注入与仿真步 ───────────────────────────────────────────────────────────

    def inject_packet(self, src: Tuple[int, int], dst: Tuple[int, int]) -> bool:
        """向网络注入一个数据包，缓冲区满则丢弃"""
        if src == dst:
            return False
        router = self.get_router(src)
        if router.can_accept():
            pkt = Packet(src=src, dst=dst, inject_cycle=self.cycle, current_pos=src)
            router.buffer.append(pkt)
            return True
        self.dropped += 1
        return False

    def step(
        self,
        routing_algo: str = 'xy',
        hotspot_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        执行一个仿真周期：
        1. 从所有路由器收集数据包
        2. 为每个包做路由决策，尝试移动到下一跳
        3. 记录状态快照并返回
        """
        self.cycle += 1

        # 步骤1：每个路由器本周期只处理一个包（限速，产生真实背压和拥塞）
        to_process: List[Tuple[Tuple[int, int], Packet]] = []
        for x in range(self.size):
            for y in range(self.size):
                router = self.routers[x][y]
                if router.buffer:
                    pkt = router.buffer.popleft()   # 取队首一个包
                    to_process.append(((x, y), pkt))

        # 步骤2：路由决策（收集后批量处理，防止同周期重复移动）
        for (x, y), pkt in to_process:
            router = self.routers[x][y]

            # 到达目的地
            if pkt.current_pos == pkt.dst:
                pkt.arrival_cycle = self.cycle
                self.delivered.append(pkt)
                router.packets_received += 1
                continue

            # 跳数超限（防死锁保护）
            if pkt.hops >= self.max_hops:
                self.dropped += 1
                continue

            # 选路
            came_from = pkt.path[-2] if len(pkt.path) >= 2 else None
            if routing_algo == 'xy':
                next_pos = self.xy_next_hop(pkt.current_pos, pkt.dst)
            elif routing_algo == 'odd_even':
                next_pos = self.odd_even_next_hop(pkt.current_pos, pkt.dst, came_from)
            else:  # ml_adaptive
                next_pos = self.ml_adaptive_next_hop(pkt.current_pos, pkt.dst, hotspot_probs)

            if next_pos is None:
                # 路由决策认为已到达
                pkt.arrival_cycle = self.cycle
                self.delivered.append(pkt)
                router.packets_received += 1
                continue

            next_router = self.get_router(next_pos)
            if next_router.can_accept():
                link_key = (pkt.current_pos, next_pos)
                self.link_usage[link_key] = self.link_usage.get(link_key, 0) + 1
                pkt.current_pos = next_pos
                pkt.hops += 1
                pkt.path.append(next_pos)
                next_router.buffer.append(pkt)
                router.packets_forwarded += 1
                self._flit_hops += 1   # 每次成功转发记一跳
            else:
                # 背压：放回队首等待
                pkt.stall_cycles += 1
                router.buffer.appendleft(pkt)

        # 步骤3：状态快照
        snapshot = self.get_state()
        self.state_history.append(snapshot)
        return snapshot

    # ── 状态与指标 ─────────────────────────────────────────────────────────────

    def get_state(self) -> np.ndarray:
        """返回全网节点利用率，shape=(size*size,)，索引 = x*size + y"""
        state = np.zeros(self.size * self.size, dtype=np.float32)
        for x in range(self.size):
            for y in range(self.size):
                state[x * self.size + y] = self.routers[x][y].utilization
        return state

    def get_hotspot_map(self, threshold: float = None) -> np.ndarray:
        thr = threshold or SimConfig.HOTSPOT_THRESHOLD
        return (self.get_state() >= thr).astype(float)

    def get_metrics(self) -> Dict:
        """计算当前累计性能指标"""
        if self.delivered:
            latencies = [p.latency for p in self.delivered if p.latency > 0]
            avg_latency = float(np.mean(latencies)) if latencies else 0.0
            max_latency = float(np.max(latencies)) if latencies else 0.0
        else:
            avg_latency = 0.0
            max_latency = 0.0

        throughput = len(self.delivered) / max(self.cycle, 1)
        utilizations = self.get_state()
        in_flight_count = sum(
            r.buffer_depth for row in self.routers for r in row
        )

        # 功耗估算（参考 Orion 2.0 NoC 功耗模型，45nm 工艺 1GHz）
        # P_dynamic = 0.6 mW × avg_flit_hops/cycle（路由器缓冲 + 交叉开关 + 链路开关）
        # P_static  = 0.5 mW × n_routers（漏电流）
        n_routers = self.size * self.size
        avg_hops_per_cycle = self._flit_hops / max(self.cycle, 1)
        power_mw = round(0.6 * avg_hops_per_cycle + 0.5 * n_routers, 3)

        return {
            'cycle': self.cycle,
            'delivered': len(self.delivered),
            'dropped': self.dropped,
            'in_flight': in_flight_count,
            'avg_latency': round(avg_latency, 2),
            'max_latency': round(max_latency, 2),
            'throughput': round(throughput, 4),
            'avg_utilization': round(float(np.mean(utilizations)), 4),
            'max_utilization': round(float(np.max(utilizations)), 4),
            'hotspot_count': int(np.sum(utilizations >= SimConfig.HOTSPOT_THRESHOLD)),
            'power_mw': power_mw,
        }

    def reset(self) -> None:
        self.__init__(self.size, self.buffer_capacity)
