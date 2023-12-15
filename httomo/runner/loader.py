from pathlib import Path
from typing import Any, Dict, Literal, Protocol, Tuple

import h5py
from mpi4py import MPI
from mpi4py.MPI import Comm

from httomo.data.hdf._utils.chunk import get_data_shape_and_offset
from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import Pattern, _get_slicing_dim


import os

from httomo.runner.backend_wrapper import BackendWrapper


class LoaderInterface(Protocol):
    """Interface to a loader object"""

    pattern: Pattern = Pattern.all
    reslice: bool = False
    method_name: str
    package_name: str = 'httomo'

    def load(self) -> DataSet:
        ...  # pragma: no cover

    def get_side_output(self) -> Dict[str, Any]:
        ...  # pragma: no cover

    @property
    def detector_x(self) -> int:
        ...  # pragma: no cover

    @property
    def detector_y(self) -> int:
        ...  # pragma: no cover


class Loader(BackendWrapper, LoaderInterface):
    """Using BackendWrapper for convenience only - it has all the logic
    for loading a method and finding all the parameters, etc.
    """

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
        self._detector_x = 0
        self._detector_y = 0

    def execute(self, dataset: DataSet) -> DataSet:
        raise NotImplementedError("Cannot execute a loader - please call load")

    def load(self) -> DataSet:
        args = self._build_kwargs(self._transform_params(self._config_params))
        ret: LoaderData = self.method(**args)
        dataset = self._process_loader_data(ret)
        dataset = self._postprocess_data(dataset)
        return dataset

    def _process_loader_data(self, ret: LoaderData) -> DataSet:
        full_shape, start_indices = get_data_shape_and_offset(ret.data, _get_slicing_dim(self.pattern) - 1, self.comm)
        dataset = DataSet(
            data=ret.data, angles=ret.angles, flats=ret.flats, darks=ret.darks,
            global_index=start_indices,
            global_shape=full_shape
        )
        self._detector_x = ret.detector_x
        self._detector_y = ret.detector_y
        return dataset

    @property
    def detector_x(self) -> int:
        return self._detector_x

    @property
    def detector_y(self) -> int:
        return self._detector_y


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


class StandardTomoLoader(DataSetSource):
    """
    Loads an individual block at a time from raw data instead of an entire chunk.
    """
    def __init__(
        self,
        in_file: Path,
        data_path: str,
        slicing_dim: Literal[0, 1, 2],
        comm: MPI.Comm,
    ) -> None:
        self._slicing_dim = slicing_dim
        self._global_shape = self._get_global_data_shape(
            comm,
            in_file,
            data_path,
        )

        chunk_index_slicing_dim = self._calculate_chunk_index_slicing_dim(
            comm.rank,
            comm.size,
            self._global_shape[slicing_dim],
        )
        next_process_chunk_index_slicing_dim = self._calculate_chunk_index_slicing_dim(
            comm.rank + 1,
            comm.size,
            self._global_shape[slicing_dim],
        )

        self._chunk_index = self._calculate_chunk_index(chunk_index_slicing_dim)
        self._chunk_shape = self._calculate_chunk_shape(
            chunk_index_slicing_dim,
            next_process_chunk_index_slicing_dim,
            self._global_shape,
        )

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._global_shape

    def _get_global_data_shape(
        self,
        comm: MPI.Comm,
        in_file: Path,
        data_path: str,
    ) -> Tuple[int, int, int]:
        with h5py.File(in_file, "r", driver="mpio", comm=comm) as f:
            dataset: h5py.Dataset = f[data_path]
            global_shape = dataset.shape
        return global_shape

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        return self._chunk_index

    def _calculate_chunk_index_slicing_dim(
        self,
        rank: int,
        nprocs: int,
        slicing_dim_length: int
    ) -> int:
        """
        Calculate the index of the chunk that is associated with the MPI process in the slicing
        dimension
        """
        return round((slicing_dim_length / nprocs) * rank)

    # TODO: Assume projection slice dim for now, and therefore assume chunk index element
    # ordering
    # TODO: Assume no previewing/cropping
    def _calculate_chunk_index(
        self,
        chunk_index_slicing_dim: int,
    ) -> Tuple[int, int, int]:
        return (chunk_index_slicing_dim, 0, 0)

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    # TODO: Assume projection slice dim for now, and therefore assume chunk shape element
    # ordering
    # TODO: Assume no previewing/cropping
    def _calculate_chunk_shape(
        self,
        current_proc_chunk_index: int,
        next_proc_chunk_index: int,
        global_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        return (
            next_proc_chunk_index - current_proc_chunk_index,
            global_shape[1],
            global_shape[2],
        )
