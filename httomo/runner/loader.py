from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Protocol, Tuple, TypeAlias, Union

import h5py
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm

from httomo.data.hdf._utils.chunk import get_data_shape_and_offset
from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet, DataSetBlock, FullFileDataSet
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


class DarksFlatsFileConfig(NamedTuple):
    file: Path
    data_path: str

class RawAngles(NamedTuple):
    data_path: str

class UserDefinedAngles(NamedTuple):
    start_angle: int
    stop_angle: int
    angles_total: int

AnglesConfig: TypeAlias = Union[RawAngles, UserDefinedAngles]


class StandardTomoLoader(DataSetSource):
    """
    Loads an individual block at a time from raw data instead of an entire chunk.
    """
    def __init__(
        self,
        in_file: Path,
        data_path: str,
        image_key_path: Optional[str],
        darks: Optional[DarksFlatsFileConfig],
        flats: Optional[DarksFlatsFileConfig],
        angles: AnglesConfig,
        slicing_dim: Literal[0, 1, 2],
        comm: MPI.Comm,
    ) -> None:
        self._in_file = in_file
        self._data_path = data_path
        self._image_key_path = image_key_path
        self._darks_config = darks
        self._flats_config = flats
        self._angles = angles
        self._slicing_dim = slicing_dim
        self._comm = comm

        self._data_indices = self._get_data_indices()
        self._global_shape = self._get_global_data_shape()

        chunk_index_slicing_dim = self._calculate_chunk_index_slicing_dim(
            comm.rank,
            comm.size,
        )
        next_process_chunk_index_slicing_dim = self._calculate_chunk_index_slicing_dim(
            comm.rank + 1,
            comm.size,
        )

        self._chunk_index = self._calculate_chunk_index(chunk_index_slicing_dim)
        self._chunk_shape = self._calculate_chunk_shape(
            chunk_index_slicing_dim,
            next_process_chunk_index_slicing_dim,
        )

        angles_arr = self._get_angles()
        darks, flats = self._get_darks_flats()

        dataset: h5py.Dataset = self._get_h5py_dataset()
        self._data = FullFileDataSet(
            data=dataset,
            angles=angles_arr,
            flats=flats,
            darks=darks,
            global_index=self._chunk_index,
            chunk_shape=self._chunk_shape,
        )

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._global_shape

    def _get_global_data_shape(self) -> Tuple[int, int, int]:
        with h5py.File(self._in_file, "r", driver="mpio", comm=self._comm) as f:
            dataset: h5py.Dataset = f[self._data_path]
            global_shape = dataset.shape
        return (len(self._data_indices), global_shape[1], global_shape[2])

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        return self._chunk_index

    def _calculate_chunk_index_slicing_dim(
        self,
        rank: int,
        nprocs: int,
    ) -> int:
        """
        Calculate the index of the chunk that is associated with the MPI process in the slicing
        dimension, taking potential darks/flats into account
        """
        shift = round((len(self._data_indices) / nprocs) * rank)
        return self._data_indices[0] + shift

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
    ) -> Tuple[int, int, int]:
        return (
            next_proc_chunk_index - current_proc_chunk_index,
            self._global_shape[1],
            self._global_shape[2],
        )

    def _get_h5py_dataset(self) -> h5py.Dataset:
        """
        Get an h5py `Dataset` object that represents the data being loaded
        """
        f = h5py.File(self._in_file, "r", driver="mpio", comm=self._comm)
        dataset: h5py.Dataset = f[self._data_path]
        return dataset

    def read_block(self, start: int, length: int) -> DataSetBlock:
        block = self._data.make_block(self._slicing_dim, start, length)
        return block

    # NOTE: This method is largely copied from `load.get_data_indices()`; that function should
    # be removed in the future if/when `StandardTomoLoader` gets merged.
    def _get_data_indices(self) -> List[int]:
        with h5py.File(self._in_file, "r", driver="mpio", comm=self._comm) as f:
            if self._image_key_path is None:
                data: h5py.Dataset = f[self._data_path]
                return np.arange(data.shape[0]).tolist()

            return np.where(f[self._image_key_path][:] == 0)[0].tolist()

    def _get_angles(self,) -> np.ndarray:
        if isinstance(self._angles, UserDefinedAngles):
            return np.linspace(
                self._angles.start_angle,
                self._angles.stop_angle,
                self._angles.angles_total,
            )

        with h5py.File(self._in_file, "r", driver="mpio", comm=self._comm) as f:
            return f[self._angles.data_path][...]

    # NOTE: This method is a barebones version of `load.get_darks_flats_together()`. Additions
    # will be necessary to support more complicated ways of specifying darks/flats.
    def _get_darks_flats(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._darks_config is not None and self._flats_config is not None:
            darks = self._get_darks_flats_separate(self._darks_config)
            flats = self._get_darks_flats_separate(self._flats_config)
            return darks, flats

        return self._get_darks_flats_together()

    def _get_darks_flats_separate(
        self,
        config: DarksFlatsFileConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(config.file, "r", driver="mpio", comm=self._comm) as f:
            return f[config.data_path][...]

    def _get_darks_flats_together(self) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self._in_file, "r", driver="mpio", comm=self._comm) as f:
            darks_indices = np.where(f[self._image_key_path][:] == 2)[0]
            flats_indices = np.where(f[self._image_key_path][:] == 1)[0]
            dataset: h5py.Dataset = f[self._data_path]
            darks = dataset[darks_indices[0]: darks_indices[-1] + 1, :, :]
            flats = dataset[flats_indices[0]: flats_indices[-1] + 1, :, :]

        return darks, flats


class DarksFlatsFileConfig2(NamedTuple):
    file: Path
    data_path: str
    image_key_path: Optional[str]


def get_darks_flats(
    darks_config: DarksFlatsFileConfig2,
    flats_config: DarksFlatsFileConfig2,
    comm: MPI.Comm,
) -> Tuple[np.ndarray, np.ndarray]:
    def get_together():
        with h5py.File(darks_config.file, "r", driver="mpio", comm=comm) as f:
            darks_indices = np.where(f[darks_config.image_key_path][:] == 2)[0]
            flats_indices = np.where(f[flats_config.image_key_path][:] == 1)[0]
            dataset: h5py.Dataset = f[darks_config.data_path]
            darks = dataset[darks_indices[0]: darks_indices[-1] + 1, :, :]
            flats = dataset[flats_indices[0]: flats_indices[-1] + 1, :, :]
        return darks, flats

    def get_separate(config: DarksFlatsFileConfig2):
        with h5py.File(config.file, "r", driver="mpio", comm=comm) as f:
            return f[config.data_path][...]

    if darks_config.file != flats_config.file:
        darks = get_separate(darks_config)
        flats = get_separate(flats_config)
        return darks, flats

    return get_together()
