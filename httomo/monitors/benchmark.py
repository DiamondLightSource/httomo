import csv
from typing import Dict, List, TextIO, Tuple
from httomo.runner.monitoring_interface import MonitoringInterface
from mpi4py import MPI
from collections import OrderedDict


class BenchmarkMonitoring(MonitoringInterface):
    def __init__(self, comm: MPI.Comm) -> None:
        self._comm = comm
        self._data: List[Dict[str, str]] = []
        self._total = 0.0

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
        self._data.append(
            OrderedDict(
                [
                    ("Type", "method"),
                    ("Rank", str(self._comm.rank)),
                    ("Name", method_name),
                    ("Task id", task_id),
                    ("Module", module_path),
                    ("Slicing dim", str(slicing_dim)),
                    ("Block offset (chunk)", str(block_idx_chunk[slicing_dim])),
                    ("Block offset (global)", str(block_idx_global[slicing_dim])),
                    ("Block dim z", str(block_dims[0])),
                    ("Block dim y", str(block_dims[1])),
                    ("Block dim x", str(block_dims[2])),
                    ("CPU time", str(cpu_time)),
                    ("GPU kernel time", str(gpu_kernel_time)),
                    ("GPU H2D time", str(gpu_h2d_time)),
                    ("GPU D2H time", str(gpu_d2h_time)),
                ]
            )
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
        self._data.append(
            OrderedDict(
                [
                    ("Type", "source"),
                    ("Rank", str(self._comm.rank)),
                    ("Name", name),
                    ("Task id", first_task_id),
                    ("Module", ""),
                    ("Slicing dim", str(slicing_dim)),
                    ("Block offset (chunk)", str(block_idx_chunk[slicing_dim])),
                    ("Block offset (global)", str(block_idx_global[slicing_dim])),
                    ("Block dim z", str(block_dims[0])),
                    ("Block dim y", str(block_dims[1])),
                    ("Block dim x", str(block_dims[2])),
                    ("CPU time", str(cpu_time)),
                    ("GPU kernel time", "0.0"),
                    ("GPU H2D time", "0.0"),
                    ("GPU D2H time", "0.0"),
                ]
            )
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
        self._data.append(
            OrderedDict(
                [
                    ("Type", "sink"),
                    ("Rank", str(self._comm.rank)),
                    ("Name", name),
                    ("Task id", last_task_id),
                    ("Module", ""),
                    ("Slicing dim", str(slicing_dim)),
                    ("Block offset (chunk)", str(block_idx_chunk[slicing_dim])),
                    ("Block offset (global)", str(block_idx_global[slicing_dim])),
                    ("Block dim z", str(block_dims[0])),
                    ("Block dim y", str(block_dims[1])),
                    ("Block dim x", str(block_dims[2])),
                    ("CPU time", str(cpu_time)),
                    ("GPU kernel time", "0.0"),
                    ("GPU H2D time", "0.0"),
                    ("GPU D2H time", "0.0"),
                ]
            )
        )

    def report_total_time(self, cpu_time: float):
        self._data.append(
            OrderedDict(
                [
                    ("Type", "total"),
                    ("Rank", str(self._comm.rank)),
                    ("Name", ""),
                    ("Task id", ""),
                    ("Module", ""),
                    ("Slicing dim", ""),
                    ("Block offset (chunk)", ""),
                    ("Block offset (global)", ""),
                    ("Block dim z", ""),
                    ("Block dim y", ""),
                    ("Block dim x", ""),
                    ("CPU time", str(cpu_time)),
                    ("GPU kernel time", "0.0"),
                    ("GPU H2D time", "0.0"),
                    ("GPU D2H time", "0.0"),
                ]
            )
        )

    def write_results(self, dest: TextIO):
        self._aggregate_mpi()
        if self._comm.rank == 0:
            writer = csv.DictWriter(dest, fieldnames=self._data[0].keys())
            writer.writeheader()
            writer.writerows(self._data)

    def _aggregate_mpi(self):
        alldata = self._comm.gather(self._data)
        if self._comm.rank == 0:
            self._data = sum(alldata, [])
