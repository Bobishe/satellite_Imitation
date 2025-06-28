"""
— Узлы (Node) имеют фиксированные координаты.
— Пакеты генерируются по пуассоновскому процессу (PoissonGenerator).
— Канал (Channel) добавляет геометрическую и дополнительную задержку,
  может терять пакеты.
— Simulation ведёт очередь событий (heapq) и собирает метрики:
  доля потерь, средняя задержка, дисперсия.
"""

import csv
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
from collections import defaultdict

C_MS = 299.792458

# ---------- базовые структуры -------------------------------------------------
@dataclass
class Node:
    node_id: int
    lat_deg: float
    lon_deg: float
    alt_km: float


@dataclass
class Packet:
    src: int
    dst: int
    created_ms: float
    size_bytes: int = 64


@dataclass(order=True)
class Event:
    time_ms: float
    priority: int
    action: Callable = field(compare=False)

# ---------- каналы ------------------------------------------------------------
class Channel:
    def __init__(self,
                 a: Node,
                 b: Node,
                 extra_delay_ms: float = 0.0,
                 loss_prob: float = 0.0,
                 call_loss_prob: float = 0.0,
                 rate_pps: float | None = None,
                 queue_limit: int | None = None):
        self.a, self.b = a, b
        self.extra_delay_ms = extra_delay_ms
        self.loss_prob = loss_prob
        self.call_loss_prob = call_loss_prob
        self.rate_pps = rate_pps
        self.queue_limit = queue_limit

        self.distance_km = _haversine_km(a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg) + abs(a.alt_km - b.alt_km)
        self.geo_delay_ms = self.distance_km / C_MS

        self.busy_until_ms = 0.0
        self.service_ms = 1000.0 / rate_pps if rate_pps else 0.0

    def transmit_delay(self, now_ms: float) -> float:
        queue_delay = 0.0
        if self.rate_pps:
            queue_delay = max(self.busy_until_ms - now_ms, 0.0)
            start_ms = now_ms + queue_delay
            self.busy_until_ms = start_ms + self.service_ms
        return queue_delay + self.service_ms + self.geo_delay_ms + self.extra_delay_ms

    def _queue_overflow(self, now_ms: float) -> bool:
        if self.queue_limit is None or not self.rate_pps:
            return False
        backlog_ms = max(self.busy_until_ms - now_ms, 0.0)
        backlog_pkts = math.ceil(backlog_ms / self.service_ms) if self.service_ms else 0
        return backlog_pkts >= self.queue_limit

    def is_lost(self) -> bool:
        return random.random() < self.loss_prob

    def is_call_lost(self) -> bool:
        return random.random() < self.call_loss_prob

# ---------- генератор трафика -------------------------------------------------
class PoissonGenerator:
    def __init__(self, rate_pps: float, src: Node, dst: Node, sim: "Simulation"):
        self.rate_pps, self.src, self.dst, self.sim = rate_pps, src, dst, sim
        self._schedule_next(self.sim.now)

    def _schedule_next(self, t_ms: float):
        interval_ms = random.expovariate(self.rate_pps) * 1000.0  
        self.sim.schedule(t_ms + interval_ms, 1, self._generate)

    def _generate(self):
        pkt = Packet(self.src.node_id, self.dst.node_id, self.sim.now)
        self.sim.on_packet_created(pkt)
        self._schedule_next(self.sim.now)

# ---------- генератор вызовов -------------------------------------------------
class CallGenerator:
    def __init__(self, rate_cps: float, src: Node, dst: Node, sim: "Simulation"):
        self.rate_cps, self.src, self.dst, self.sim = rate_cps, src, dst, sim
        self._schedule_next(self.sim.now)

    def _schedule_next(self, t_ms: float):
        interval_ms = random.expovariate(self.rate_cps) * 1000.0
        self.sim.schedule(t_ms + interval_ms, 1, self._attempt)

    def _attempt(self):
        self.sim.on_call_attempt(self.src.node_id, self.dst.node_id)
        self._schedule_next(self.sim.now)

# ---------- симулятор ---------------------------------------------------------
class Simulation:
    def __init__(self, nodes: List[Node], channels: Dict[Tuple[int, int], Channel], end_time_ms: float,
                 metrics_fname: str = "metrics.csv"):
        self.nodes = {n.node_id: n for n in nodes}
        self.channels = channels
        self.end_time_ms = end_time_ms
        self.metrics_fname = metrics_fname
        self.event_queue: List[Event] = []
        self.now = 0.0

        self.graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (a, b), ch in self.channels.items():
            weight = ch.geo_delay_ms + ch.extra_delay_ms
            self.graph[a].append((b, weight))

        # метрики
        self.sent = self.delivered = self.lost = 0
        self.delay_sum = self.delay_sq_sum = 0.0
        # call metrics
        self.call_attempts = self.call_failures = 0

    # API для генераторов/каналов
    def on_packet_created(self, pkt: Packet):
        self.sent += 1
        path = self._find_best_path(pkt.src, pkt.dst)
        if not path:
            self.lost += 1
            return
        self._send_packet(pkt, path, 0, 0.0)

    def _forward(self, pkt: Packet, ch: Channel):
        if ch.is_lost() or ch._queue_overflow(self.now):
            self.lost += 1
            return
        delay = ch.transmit_delay(self.now)
        self.schedule(self.now + delay, 0, lambda: self._on_delivered(pkt, delay))

    def _send_packet(self, pkt: Packet, path: List[int], index: int, acc_delay: float):
        if index == len(path) - 1:
            self._on_delivered(pkt, acc_delay)
            return
        ch = self.channels[(path[index], path[index + 1])]
        if ch.is_lost() or ch._queue_overflow(self.now):
            self.lost += 1
            return
        delay = ch.transmit_delay(self.now)
        self.schedule(
            self.now + delay,
            0,
            lambda: self._send_packet(pkt, path, index + 1, acc_delay + delay),
        )

    def _on_delivered(self, pkt: Packet, delay_ms: float):
        self.delivered += 1
        self.delay_sum += delay_ms
        self.delay_sq_sum += delay_ms ** 2

    def _find_best_path(self, src: int, dst: int) -> List[int] | None:
        queue: List[Tuple[float, int]] = [(0.0, src)]
        dist: Dict[int, float] = {src: 0.0}
        prev: Dict[int, int] = {}
        while queue:
            d, u = heapq.heappop(queue)
            if u == dst:
                break
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self.graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(queue, (nd, v))
        if dst not in dist:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    # обработка попытки вызова
    def on_call_attempt(self, src_id: int, dst_id: int):
        self.call_attempts += 1
        path = self._find_best_path(src_id, dst_id)
        if not path:
            self.call_failures += 1
            return
        for i in range(len(path) - 1):
            ch = self.channels[(path[i], path[i + 1])]
            if ch.is_call_lost():
                self.call_failures += 1
                return

    # движок событий
    def schedule(self, time_ms: float, priority: int, action: Callable):
        heapq.heappush(self.event_queue, Event(time_ms, priority, action))

    def run(self):
        while self.event_queue and self.now <= self.end_time_ms:
            ev = heapq.heappop(self.event_queue)
            self.now = ev.time_ms
            if self.now > self.end_time_ms:
                break
            ev.action()
        self._write_metrics()

    # вывод метрик
    def _write_metrics(self):
        delivered = max(self.delivered, 1)
        avg_delay = self.delay_sum / delivered
        var_delay = (self.delay_sq_sum / delivered) - avg_delay ** 2
        loss_rate = self.lost / max(self.sent, 1)
        call_loss_rate = self.call_failures / max(self.call_attempts, 1)
        with open(self.metrics_fname, "w", newline="") as f:
            csv.writer(f).writerow(
                ["total_sent", "delivered", "lost", "loss_rate", "avg_delay_ms",
                 "delay_var", "call_attempts", "call_failures", "call_loss_rate"])
            csv.writer(f).writerow(
                [self.sent, self.delivered, self.lost, f"{loss_rate:.6f}",
                 f"{avg_delay:.3f}", f"{var_delay:.3f}",
                 self.call_attempts, self.call_failures,
                 f"{call_loss_rate:.6f}"])
        print(f"Metrics written to {self.metrics_fname}")

# ---------- утилиты -----------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))  # км


def load_config(path: str) -> dict:
    """Load simulation configuration from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_simulation(cfg: dict) -> Simulation:
    """Build Simulation instance from configuration dictionary."""
    nodes: List[Node] = []
    node_map: Dict[int, Node] = {}
    for n in cfg.get("nodes", []):
        node = Node(n["id"], n["lat"], n["lon"], n["alt"])
        nodes.append(node)
        node_map[node.node_id] = node

    channels: Dict[Tuple[int, int], Channel] = {}
    for ch in cfg.get("channels", []):
        a = node_map[ch["src"]]
        b = node_map[ch["dst"]]
        channels[(a.node_id, b.node_id)] = Channel(
            a,
            b,
            extra_delay_ms=ch.get("extra_delay_ms", 0.0),
            loss_prob=ch.get("loss_prob", 0.0),
            call_loss_prob=ch.get("call_loss_prob", 0.0),
            rate_pps=ch.get("rate_pps"),
            queue_limit=ch.get("queue_limit"),
        )

    sim = Simulation(
        nodes,
        channels,
        end_time_ms=cfg.get("end_time_ms", 0),
        metrics_fname=cfg.get("metrics_file", "metrics.csv"),
    )

    for gen in cfg.get("packet_generators", []):
        PoissonGenerator(gen["rate_pps"], node_map[gen["src"]], node_map[gen["dst"]], sim)

    for gen in cfg.get("call_generators", []):
        CallGenerator(gen["rate_cps"], node_map[gen["src"]], node_map[gen["dst"]], sim)

    return sim

# ---------- демонстрация ------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simulation from config file")
    parser.add_argument("--config", default="config.json", help="Path to JSON configuration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = build_simulation(cfg)
    sim.run()
