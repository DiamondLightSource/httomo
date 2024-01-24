from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.dataset import DataSetBlock
from httomo.runner.methods_repository_interface import MethodRepository

from mpi4py.MPI import Comm
from typing import Dict


class DezingingWrapper(GenericMethodWrapper):
    """Wraps the remove_outlier3d method, to clean/dezing the data.
    Note that this method is applied to all elements of the dataset, i.e.
    data, darks, and flats.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return method_name == "remove_outlier3d"

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
        assert (
            method_name == "remove_outlier3d"
        ), "Only remove_outlier3d is supported at the moment"
        self._flats_darks_processed = False

    def execute(self, dataset: DataSetBlock) -> DataSetBlock:
        # check if data needs to be transfered host <-> device
        dataset = self._transfer_data(dataset)

        dataset.data = self.method(dataset.data, **self._config_params)
        if not self._flats_darks_processed:
            darks = self.method(dataset.darks, **self._config_params)
            flats = self.method(dataset.flats, **self._config_params)
            ds = dataset.base
            ds.unlock()
            ds.darks = darks
            ds.flats = flats
            ds.lock()
            self._flats_darks_processed = True

        return dataset
