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
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

C_MS = 299_792.458  

# ---------- базовые структуры -------------------------------------------------
@dataclass
class Node:
    node_id: int
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0


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
    def __init__(self, a: Node, b: Node, extra_delay_ms: float = 0.0, loss_prob: float = 0.0):
        self.a, self.b = a, b
        self.extra_delay_ms = extra_delay_ms
        self.loss_prob = loss_prob
        self.distance_km = _haversine_km(a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg) + abs(a.alt_km - b.alt_km)
        self.geo_delay_ms = self.distance_km / C_MS

    def transmit_delay(self) -> float:
        return self.geo_delay_ms + self.extra_delay_ms

    def is_lost(self) -> bool:
        return random.random() < self.loss_prob

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

# ---------- симулятор ---------------------------------------------------------
class Simulation:
    def __init__(self, nodes: List[Node], channels: Dict[Tuple[int, int], Channel], end_time_ms: float):
        self.nodes = {n.node_id: n for n in nodes}
        self.channels = channels
        self.end_time_ms = end_time_ms
        self.event_queue: List[Event] = []
        self.now = 0.0

        # метрики
        self.sent = self.delivered = self.lost = 0
        self.delay_sum = self.delay_sq_sum = 0.0

    # API для генераторов/каналов
    def on_packet_created(self, pkt: Packet):
        self.sent += 1
        self._forward(pkt, self.channels[(pkt.src, pkt.dst)])

    def _forward(self, pkt: Packet, ch: Channel):
        if ch.is_lost():
            self.lost += 1
            return
        delay = ch.transmit_delay()
        self.schedule(self.now + delay, 0, lambda: self._on_delivered(pkt, delay))

    def _on_delivered(self, pkt: Packet, delay_ms: float):
        self.delivered += 1
        self.delay_sum += delay_ms
        self.delay_sq_sum += delay_ms ** 2

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
    def _write_metrics(self, fname: str = "metrics.csv"):
        delivered = max(self.delivered, 1)
        avg_delay = self.delay_sum / delivered
        var_delay = (self.delay_sq_sum / delivered) - avg_delay ** 2
        loss_rate = self.lost / max(self.sent, 1)
        with open(fname, "w", newline="") as f:
            csv.writer(f).writerow(
                ["total_sent", "delivered", "lost", "loss_rate", "avg_delay_ms", "delay_var"])
            csv.writer(f).writerow(
                [self.sent, self.delivered, self.lost, f"{loss_rate:.6f}", f"{avg_delay:.3f}", f"{var_delay:.3f}"])
        print(f"Metrics written to {fname}")

# ---------- утилиты -----------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))  # км

# ---------- демонстрация ------------------------------------------------------
if __name__ == "__main__":
    # Два узла: Москва → Санкт-Петербург, канал 2 мс + 1 % потерь, 10 pkt/s, 30 с
    n0 = Node(0, 55.751244, 37.618423)
    n1 = Node(1, 59.93106, 30.36057)
    ch = Channel(n0, n1, extra_delay_ms=2.0, loss_prob=0.01)
    sim = Simulation([n0, n1], {(0, 1): ch}, end_time_ms=30_000.0)
    PoissonGenerator(rate_pps=10, src=n0, dst=n1, sim=sim)
    sim.run()
