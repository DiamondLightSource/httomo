from httomo.block_interfaces import T
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import catchtime, log_rank, xp, gpu_enabled


from mpi4py.MPI import Comm
from mpi4py import MPI

from typing import Any, Dict, Optional, Tuple


class StatsCalcWrapper(GenericMethodWrapper):
    """This class calculates global statistics and deliver a side_output.
    It also forces to return the original dataset to be passed to the next method.

    Note that the side output is only set once the last block in the chunk has been
    processed.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return method_name == "calculate_stats"

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        save_result: Optional[bool] = None,
        output_mapping: Dict[str, str] = {},
        **kwargs,
    ):
        super().__init__(
            method_repository,
            module_path,
            method_name,
            comm,
            save_result,
            output_mapping,
            **kwargs,
        )
        self._min = float("inf")
        self._max = float("-inf")
        self._sum: float = 0.0
        self._elements: int = 0

    def _transfer_data(self, dataset: T) -> T:
        # don't transfer anything (either way) at this point
        return dataset

    def _run_method(self, dataset: T, args: Dict[str, Any]) -> T:
        # transfer data to GPU if we can / have it available (always faster),
        # but don't want to fail if we don't have a GPU (underlying method works for both)
        # and don't touch original dataset
        if gpu_enabled and dataset.is_cpu:
            with catchtime() as t:
                args[self._parameters[0]] = xp.asarray(dataset.data)
            self._gpu_time_info.host2device += t.elapsed
        ret = self._method(**args)
        return self._process_return_type(ret, dataset)

    def _process_return_type(self, ret: Any, input_block: T) -> T:
        assert isinstance(ret, tuple), "expected return type is a tuple"
        assert len(ret) == 4, "A 4-tuple of stats values is expected"

        self._min = min(self._min, float(ret[0]))
        self._max = max(self._max, float(ret[1]))
        self._sum += float(ret[2])
        self._elements += int(ret[3])

        if input_block.is_last_in_chunk:
            glob_stats = self._accumulate_chunks()
            stats_str = f"Global min {glob_stats[0]}, Global max {glob_stats[1]}, Global mean {glob_stats[2]}"
            log_rank(stats_str, comm=self.comm)
            self._side_output["glob_stats"] = glob_stats

        return input_block

    def _accumulate_chunks(self) -> Tuple[float, float, float, int]:
        # reduce to rank 0 process
        min_glob = self.comm.reduce(float(self._min), MPI.MIN)
        max_glob = self.comm.reduce(float(self._max), MPI.MAX)
        sum_glob = self.comm.reduce(float(self._sum), MPI.SUM)
        elem_glob = self.comm.reduce(int(self._elements), MPI.SUM)

        glob_stats: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
        if self.comm.rank == 0:
            assert isinstance(min_glob, float)
            assert isinstance(max_glob, float)
            assert isinstance(sum_glob, float)
            assert isinstance(elem_glob, int)

            # calculate (min, max, mean, total_elements) tuple
            glob_stats = (
                min_glob,
                max_glob,
                sum_glob / elem_glob,
                elem_glob,
            )

        # broadcast result
        glob_stats = self.comm.bcast(glob_stats)

        return glob_stats
