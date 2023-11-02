from typing import Any, Protocol
from mpi4py import MPI
from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import Pattern


import os

from httomo.runner.backend_wrapper import BackendWrapper


class LoaderInterface(Protocol):
    """Interface to a loader object"""

    pattern: Pattern
    reslice: bool

    def load(self) -> DataSet:
        ...


class Loader(LoaderInterface, BackendWrapper):
    """Using BackendWrapper for convenience only - it has all the logic
    for loading a method and finding all the parameters, etc.
    """

    def execute(self, dataset: DataSet) -> DataSet:
        raise NotImplementedError("Cannot execute a loader - please call load")

    def load(self) -> DataSet:
        args = self._build_kwargs(self._transform_params(self._config_params))
        ret: LoaderData = self.method(**args)
        dataset = self._process_loader_data(ret)
        dataset = self._postprocess_data(dataset)
        return dataset

    def _process_loader_data(self, ret: LoaderData) -> DataSet:
        dataset = DataSet(
            data=ret.data, angles=ret.angles, flats=ret.flats, darks=ret.darks
        )
        return dataset


def make_loader(
    method_repository: MethodRepository,
    module_path: str,
    method_name: str,
    comm: MPI.Comm,
    **kwargs,
) -> Loader:
    """Factory function to generate the appropriate wrapper based on the module
    path and method name for loaders.

    Parameters
    ----------

    method_repository: MethodRepository
        Repository of methods that we can use the query properties
    module_path: str
        Path to the module where the method is in python notation, e.g. "httomolibgpu.prep.normalize"
    method_name: str
        Name of the method (function within the given module)
    comm: Comm
        MPI communicator object
    kwargs:
        Arbitrary keyword arguments that get passed to the method as parameters.

    Returns
    -------

    Loader
        An instance of a loader class (which is also a BackendWrapper)
    """
    
    # note: once we have different kinds of loaders, this function can 
    # be used like the make_backend_wrapper factory function

    return Loader(
        method_repository=method_repository,
        module_path=module_path,
        method_name=method_name,
        comm=comm,
        **kwargs,
    )
