from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.methods import calculate_stats
from httomo.runner.dataset import DataSetBlock
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import log_rank


from mpi4py.MPI import Comm


from typing import Any, Dict


class StatsCalcWrapper(GenericMethodWrapper):
    """This class calculates global statistics and deliver a side_output.
    It also forces to return the original dataset to be passed to the next method.
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
        output_mapping: Dict[str, str] = {},
        **kwargs,
    ):
        super().__init__(
            method_repository, module_path, method_name, comm, output_mapping, **kwargs
        )

    def _run_method(self, dataset: DataSetBlock, args: Dict[str, Any]) -> DataSetBlock:
        res = calculate_stats(dataset.data, comm=self.comm)
        stats_str = f"Global min {res[0]}, Global max {res[1]}, Global mean {res[2]}"
        log_rank(stats_str, comm=self.comm)
        return self._process_return_type(res, dataset)

    def _process_return_type(
        self, ret: Any, input_dataset: DataSetBlock
    ) -> DataSetBlock:
        # getting (min, max, mean, total_elements) tuple  and assigning to side_output
        self._side_output["glob_stats"] = ret
        return input_dataset
