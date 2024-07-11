from typing import Protocol, TextIO, Tuple


class MonitoringInterface(Protocol):
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
    ): ...  # pragma: no cover

    def report_source_block(
        self,
        name: str,
        first_task_id: str,
        slicing_dim: int,
        block_dims: Tuple[int, int, int],
        block_idx_chunk: Tuple[int, int, int],
        block_idx_global: Tuple[int, int, int],
        cpu_time: float,
    ): ...  # pragma: no cover

    def report_sink_block(
        self,
        name: str,
        last_task_id: str,
        slicing_dim: int,
        block_dims: Tuple[int, int, int],
        block_idx_chunk: Tuple[int, int, int],
        block_idx_global: Tuple[int, int, int],
        cpu_time: float,
    ): ...  # pragma: no cover

    def report_total_time(self, cpu_time: float): ...  # pragma: no cover

    def write_results(self, dest: TextIO): ...  # pragma: no cover
