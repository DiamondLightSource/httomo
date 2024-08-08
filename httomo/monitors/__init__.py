from typing import Dict, List, Optional

from mpi4py import MPI

from httomo.monitors.aggregate import AggregateMonitoring
from httomo.monitors.benchmark import BenchmarkMonitoring
from httomo.monitors.summary import SummaryMonitor
from httomo.runner.monitoring_interface import MonitoringInterface


MONITORS_MAP = {"bench": BenchmarkMonitoring, "summary": SummaryMonitor}


def make_monitors(
    monitor_descriptors: List[str],
    comm: MPI.Comm,
) -> Optional[MonitoringInterface]:
    if len(monitor_descriptors) == 0:
        return None

    monitors: List[MonitoringInterface] = []
    for descriptor in monitor_descriptors:
        if descriptor not in MONITORS_MAP:
            raise ValueError(
                f"Unknown monitor '{descriptor}'. Please choose one of {MONITORS_MAP.keys()}"
            )
        monitors.append(MONITORS_MAP[descriptor](comm))

    return AggregateMonitoring(monitors)
