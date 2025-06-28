"""Satellite network simulation package."""

from .simulation import (
    Node,
    Packet,
    Channel,
    PoissonGenerator,
    CallGenerator,
    Simulation,
    load_config,
    build_simulation,
)
try:
    from .monitor import MonitoringSimulation, run_simulation, show_results, launch_ui
except Exception:  # matplotlib may be missing
    MonitoringSimulation = None
    run_simulation = None
    show_results = None
    launch_ui = None

__all__ = [
    "Node",
    "Packet",
    "Channel",
    "PoissonGenerator",
    "CallGenerator",
    "Simulation",
    "MonitoringSimulation",
    "load_config",
    "build_simulation",
    "run_simulation",
    "show_results",
    "launch_ui",
]
