"""
— расширяет базовый класс Simulation, чтобы сохранять метрики по времени;
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import simple_sim

# ── расширение симулятора ─────────────────────────────────────────────────────
class MonitoringSimulation(simple_sim.Simulation):
    def __init__(self, nodes, channels, end_time_ms, bin_ms=1000, metrics_fname="metrics.csv"):
        super().__init__(nodes, channels, end_time_ms, metrics_fname=metrics_fname)
        self.bin_ms = bin_ms                 # ширина «окна» в миллисекундах
        self.timeline = []                   # список словарей с метриками
        self._reset_bin()

    # внутренний счётчик
    def _reset_bin(self):
        self.next_edge = self.now + self.bin_ms
        self.bin_sent = self.bin_delivered = self.bin_lost = 0
        self.bin_delay_sum = 0.0

    #  фиксируем отправку
    def on_packet_created(self, pkt):
        self.bin_sent += 1
        super().on_packet_created(pkt)

    # переопределяем пересылку чтобы учитывать потерянные пакеты
    def _forward(self, pkt, ch):
        if ch.is_lost() or ch._queue_overflow(self.now):
            self.lost += 1
            self.bin_lost += 1
            return
        delay = ch.transmit_delay(self.now)
        self.schedule(self.now + delay, 0, lambda: self._on_delivered(pkt, delay))

    # доставка пакета — считаем задержку
    def _on_delivered(self, pkt, delay_ms):
        self.bin_delivered += 1
        self.bin_delay_sum += delay_ms
        super()._on_delivered(pkt, delay_ms)

    # проверяем не пора ли закрыть текущий интервал
    def _maybe_close_bin(self):
        if self.now >= self.next_edge:
            avg_delay = (self.bin_delay_sum / self.bin_delivered) if self.bin_delivered else 0.0
            loss_rate = (self.bin_lost / self.bin_sent) if self.bin_sent else 0.0
            self.timeline.append(
                {"t": self.next_edge / 1000.0,        # секунда
                 "delay": avg_delay,                  # мс
                 "loss": loss_rate})                  # 0‒1
            self._reset_bin()

    # оборачиваем каждое действие событием проверки бин-среза
    def schedule(self, time_ms, priority, action):
        def wrapped():
            action()
            self._maybe_close_bin()
        super().schedule(time_ms, priority, wrapped)


# ── запуск модели и получение временного ряда ─────────────────────────────────
def run_simulation(cfg):
    """Run monitoring simulation and return collected timeline."""
    sim = simple_sim.build_simulation(cfg)
    sim = MonitoringSimulation(
        list(sim.nodes.values()),
        sim.channels,
        end_time_ms=sim.end_time_ms,
        bin_ms=cfg.get("bin_ms", 1000),
        metrics_fname=sim.metrics_fname,
    )

    for gen in cfg.get("packet_generators", []):
        simple_sim.PoissonGenerator(gen["rate_pps"], sim.nodes[gen["src"]], sim.nodes[gen["dst"]], sim)

    for gen in cfg.get("call_generators", []):
        simple_sim.CallGenerator(gen["rate_cps"], sim.nodes[gen["src"]], sim.nodes[gen["dst"]], sim)

    sim.run()
    return sim.timeline


def show_results(timeline):
    """Display charts from collected timeline."""
    t = [p["t"] for p in timeline]
    delay = [p["delay"] for p in timeline]
    loss = [p["loss"] for p in timeline]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, delay)
    axs[0].set_ylabel("сред. задержка, мс")
    axs[0].grid(True)

    axs[1].plot(t, loss)
    axs[1].set_xlabel("время, с")
    axs[1].set_ylabel("доля потерь")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def _on_start(root, cfg):
    """Callback for the start button."""
    root.destroy()
    timeline = run_simulation(cfg)
    show_results(timeline)


def launch_ui(cfg):
    """Show network topology and start button."""
    root = tk.Tk()
    root.title("Network topology")

    fig = plt.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    nodes = cfg.get("nodes", [])
    positions = {n["id"]: (n["lon"], n["lat"]) for n in nodes}
    ax.scatter([p[0] for p in positions.values()], [p[1] for p in positions.values()])

    for ch in cfg.get("channels", []):
        a = positions[ch["src"]]
        b = positions[ch["dst"]]
        ax.plot([a[0], b[0]], [a[1], b[1]], "k--")

    ax.set_xlabel("долгота")
    ax.set_ylabel("широта")
    ax.set_title("Топология сети")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    info = f"Узлов: {len(nodes)}\nКаналов: {len(cfg.get('channels', []))}\nВремя: {cfg.get('end_time_ms',0)} мс"
    tk.Label(root, text=info).pack(pady=5)
    tk.Button(root, text="Запустить имитацию", command=lambda: _on_start(root, cfg)).pack(pady=5)

    root.mainloop()


def main(cfg_path="config.json"):
    cfg = simple_sim.load_config(cfg_path)
    launch_ui(cfg)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run monitoring simulation from config")
    parser.add_argument("--config", default="config.json", help="Path to JSON configuration")
    args = parser.parse_args()

    main(args.config)
