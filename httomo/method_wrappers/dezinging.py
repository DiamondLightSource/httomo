from httomo.block_interfaces import T
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.method_wrapper import GpuTimeInfo
from httomo.runner.methods_repository_interface import MethodRepository

from mpi4py.MPI import Comm
from typing import Dict, Optional

from httomo.utils import catch_gputime


class DezingingWrapper(GenericMethodWrapper):
    """Wraps the remove_outlier method, to clean/dezing the data.
    Note that this method is applied to all elements of the dataset, i.e.
    data, darks, and flats.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return method_name == "remove_outlier"

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
        assert (
            method_name == "remove_outlier"
        ), "Only remove_outlier is supported at the moment"
        self._flats_darks_processed = False

    def execute(self, block: T) -> T:
        self._gpu_time_info = GpuTimeInfo()
        # check if data needs to be transfered host <-> device
        block = self._transfer_data(block)

        if "axis" in self.parameters:
            args = self._build_kwargs(
                self._transform_params(self._config_params), block
            )
            # plug in the correct value for axis instead of auto. In case if axis is absent we
            # take it equal to 0. This is to avoid failure, but (missing) template parameters
            # should be tested in the yaml checker potentially
            self._config_params["axis"] = args.get("axis", 0)

        with catch_gputime() as t:
            block.data = self.method(block.data, **self._config_params)
            if not self._flats_darks_processed:
                block.darks = self.method(block.darks, **self._config_params)
                block.flats = self.method(block.flats, **self._config_params)
                self._flats_darks_processed = True

        self._gpu_time_info.kernel = t.elapsed

        return block
