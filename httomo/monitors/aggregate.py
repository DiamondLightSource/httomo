from typing import List, TextIO, Tuple
from httomo.runner.monitoring_interface import MonitoringInterface


class AggregateMonitoring(MonitoringInterface):
    def __init__(self, monitors: List[MonitoringInterface]):
        self._monitors = monitors

    def report_method_block(
        self,
        method_name: str,
        module_path: str,
        task_id: str,
        slicing_dim: int,
        block_dims: Tuple[int, int, int],
        block_idx_chunk: Tuple[int, int, int],
        block_idx_global: Tuple[int, int, int],
        cpu_time: float,
        gpu_kernel_time: float = 0.0,
        gpu_h2d_time: float = 0.0,
        gpu_d2h_time: float = 0.0,
    ):
        for m in self._monitors:
            m.report_method_block(
                method_name,
                module_path,
                task_id,
                slicing_dim,
                block_dims,
                block_idx_chunk,
                block_idx_global,
                cpu_time,
                gpu_kernel_time,
                gpu_h2d_time,
                gpu_d2h_time,
            )

    def report_source_block(
        self,
        name: str,
        first_task_id: str,
        slicing_dim: int,
        block_dims: Tuple[int, int, int],
        block_idx_chunk: Tuple[int, int, int],
        block_idx_global: Tuple[int, int, int],
        cpu_time: float,
    ):
        for m in self._monitors:
            m.report_source_block(
                name,
                first_task_id,
                slicing_dim,
                block_dims,
                block_idx_chunk,
                block_idx_global,
                cpu_time,
            )

    def report_sink_block(
        self,
        name: str,
        last_task_id: str,
        slicing_dim: int,
        block_dims: Tuple[int, int, int],
        block_idx_chunk: Tuple[int, int, int],
        block_idx_global: Tuple[int, int, int],
        cpu_time: float,
    ):
        for m in self._monitors:
            m.report_sink_block(
                name,
                last_task_id,
                slicing_dim,
                block_dims,
                block_idx_chunk,
                block_idx_global,
                cpu_time,
            )

    def report_total_time(self, cpu_time: float):
        for m in self._monitors:
            m.report_total_time(cpu_time)

    def write_results(self, dest: TextIO):
        for m in self._monitors:
            m.write_results(dest)
            dest.writelines("\n")
            dest.flush()
