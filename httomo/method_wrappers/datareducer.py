from httomo.block_interfaces import T
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository

from mpi4py.MPI import Comm
from typing import Dict, Optional


class DatareducerWrapper(GenericMethodWrapper):
    """Wraps the data_reducer method, to be applied to flats and darks only.
    The method is sequentially applied to each dataset.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return method_name == "data_reducer"

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
            method_name == "data_reducer"
        ), "Only data_reducer is supported at the moment"
        self._flats_darks_processed = False

    def execute(self, block: T) -> T:
        # check if data needs to be transfered host <-> device
        block = self._transfer_data(block)

        if not self._flats_darks_processed:
            block.darks = self.method(block.darks, **self._config_params)
            block.flats = self.method(block.flats, **self._config_params)
            self._flats_darks_processed = True
        return block
