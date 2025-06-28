"""Entry point for satellite simulation package."""
from __future__ import annotations

import argparse

from satellite import build_simulation, load_config, launch_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="Satellite network simulation")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run simulation with monitoring UI",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.monitor:
        launch_ui(cfg)
    else:
        sim = build_simulation(cfg)
        sim.run()


if __name__ == "__main__":
    main()
