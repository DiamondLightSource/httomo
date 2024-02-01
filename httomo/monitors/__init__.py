from typing import Dict, List
from httomo.monitors.aggregate import AggregateMonitoring
from httomo.monitors.benchmark import BenchmarkMonitoring
from httomo.monitors.summary import SummaryMonitor
from httomo.runner.monitoring_interface import MonitoringInterface


MONITORS_MAP = {
    "bench": BenchmarkMonitoring,
    "summary": SummaryMonitor
}

def make_monitors(monitor_descriptors: List[str]) -> MonitoringInterface:
    monitors: List[MonitoringInterface] = []
    for descriptor in monitor_descriptors:
        if descriptor not in MONITORS_MAP:
            raise ValueError(f"Unknown monitor '{descriptor}'. Please choose one of {MONITORS_MAP.keys}")
        monitors.append(MONITORS_MAP[descriptor]())
        
    return AggregateMonitoring(monitors)