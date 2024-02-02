from typing import TextIO, Tuple
from mpi4py import MPI
from httomo.runner.monitoring_interface import MonitoringInterface


class SummaryMonitor(MonitoringInterface):
    def __init__(self) -> None:
        self._methods_cpu = 0.0
        self._methods_gpu = 0.0
        self._h2d = 0.0
        self._d2h = 0.0
        self._sources = 0.0
        self._sinks = 0.0
        self._total = 0.0
        self._total_agg = 0.0
        self._comm = MPI.COMM_WORLD
    
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
        gpu_d2h_time: float = 0.0        
    ):
        self._methods_cpu += cpu_time
        self._methods_gpu += gpu_kernel_time
        self._h2d += gpu_h2d_time
        self._d2h += gpu_d2h_time
        
    def report_source_block(
        self,
        name: str,
        first_task_id: str,
        slicing_dim: int,
        block_dims: Tuple[int, int, int],
        block_idx_chunk: Tuple[int, int, int],
        block_idx_global: Tuple[int, int, int],
        cpu_time: float
    ):
        self._sources += cpu_time
        
    def report_sink_block(
        self,
        name: str,
        last_task_id: str,
        slicing_dim: int,
        block_dims: Tuple[int, int, int],
        block_idx_chunk: Tuple[int, int, int],
        block_idx_global: Tuple[int, int, int],
        cpu_time: float
    ):
        self._sinks += cpu_time
        
    def report_total_time(self, cpu_time: float):
        self._total = cpu_time
        
    def write_results(self, dest: TextIO):
        self._aggregate_mpi()
        if self._comm.rank == 0:
            dest.write('\n'.join([
                f"Summary Statistics (aggregated across {self._comm.size} processes):",
                f"  Total methods CPU time: {self._methods_cpu:>10.3f}s",
                f"  Total methods GPU time: {self._methods_gpu:>10.3f}s",
                f"  Total host2device time: {self._h2d:>10.3f}s",
                f"  Total device2host time: {self._d2h:>10.3f}s",
                f"  Total sources time    : {self._sources:>10.3f}s",
                f"  Total sinks time      : {self._sinks:>10.3f}s",
                f"  Other overheads       : {self._total_agg - sum([self._methods_cpu, self._sources, self._sinks]):>10.3f}s",
                f"  ------------------------" + "-" * 15,
                f"  Total pipeline time   : {self._total_agg:>10.3f}s",
                f"  Total wall time       : {self._total:>10.3f}s"
            ]))
        
    def _aggregate_mpi(self):
        self._total_agg = self._total
        if self._comm.size == 1:
            return
        self._methods_cpu = self._comm.reduce(self._methods_cpu, MPI.SUM)
        self._methods_gpu = self._comm.reduce(self._methods_gpu, MPI.SUM)
        self._sources = self._comm.reduce(self._sources, MPI.SUM)
        self._sinks = self._comm.reduce(self._sinks, MPI.SUM)
        self._h2d = self._comm.reduce(self._h2d, MPI.SUM)
        self._d2h = self._comm.reduce(self._d2h, MPI.SUM)
        self._total_agg = self._comm.reduce(self._total_agg, MPI.SUM)