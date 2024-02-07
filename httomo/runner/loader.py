from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    Union,
)

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

from httomo.runner.method_wrapper import MethodWrapper


class LoaderInterface(Protocol):
    """Interface to a loader object"""

    # Patterns the loader supports
    pattern: Pattern = Pattern.all
    # purely informational, for use by the logger
    method_name: str
    package_name: str = "httomo"

    def make_data_source(self) -> DataSetSource:
        """Create a dataset source that can produce blocks of data from the file.

        This will be called after the patterns and sections have been determined,
        just before the execution of the first section starts."""
        ...  # pragma: no cover

    @property
    def detector_x(self) -> int:
        """detector x-dimension of the loaded data"""
        ...  # pragma: no cover

    @property
    def detector_y(self) -> int:
        """detector y-dimension of the loaded data"""
        ...  # pragma: no cover

    @property
    def angles_total(self) -> int:
        """angles dimension of the loaded data"""
        ...  # pragma: no cover


class DarksFlatsFileConfig(NamedTuple):
    file: Path
    data_path: str
    image_key_path: Optional[str]


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
        darks: DarksFlatsFileConfig,
        flats: DarksFlatsFileConfig,
        angles: AnglesConfig,
        slicing_dim: Literal[0, 1, 2],
        comm: MPI.Comm,
    ) -> None:
        self._in_file = in_file
        self._data_path = data_path
        self._image_key_path = image_key_path
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
        darks_arr, flats_arr = get_darks_flats(darks, flats, comm)

        dataset: h5py.Dataset = self._get_h5py_dataset()
        self._data = FullFileDataSet(
            data=dataset,
            angles=angles_arr,
            flats=flats_arr,
            darks=darks_arr,
            global_index=self._chunk_index,
            chunk_shape=self._chunk_shape,
        )

    @property
    def dtype(self) -> np.dtype:
        return self._data.data.dtype

    @property
    def flats(self) -> np.ndarray:
        return self._data.flats

    @property
    def darks(self) -> np.ndarray:
        return self._data.darks

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

    def _get_angles(
        self,
    ) -> np.ndarray:
        if isinstance(self._angles, UserDefinedAngles):
            return np.linspace(
                self._angles.start_angle,
                self._angles.stop_angle,
                self._angles.angles_total,
            )

        with h5py.File(self._in_file, "r", driver="mpio", comm=self._comm) as f:
            return f[self._angles.data_path][...]

    def finalize(self):
        pass


def get_darks_flats(
    darks_config: DarksFlatsFileConfig,
    flats_config: DarksFlatsFileConfig,
    comm: MPI.Comm,
) -> Tuple[np.ndarray, np.ndarray]:
    def get_together():
        with h5py.File(darks_config.file, "r", driver="mpio", comm=comm) as f:
            darks_indices = np.where(f[darks_config.image_key_path][:] == 2)[0]
            flats_indices = np.where(f[flats_config.image_key_path][:] == 1)[0]
            dataset: h5py.Dataset = f[darks_config.data_path]
            darks = dataset[darks_indices[0] : darks_indices[-1] + 1, :, :]
            flats = dataset[flats_indices[0] : flats_indices[-1] + 1, :, :]
        return darks, flats

    def get_separate(config: DarksFlatsFileConfig):
        with h5py.File(config.file, "r", driver="mpio", comm=comm) as f:
            return f[config.data_path][...]

    if darks_config.file != flats_config.file:
        darks = get_separate(darks_config)
        flats = get_separate(flats_config)
        return darks, flats

    return get_together()


class StandardLoaderWrapper(LoaderInterface):
    def __init__(
        self,
        comm: Comm,
        # parameters that should be adjustable from YAML
        in_file: Path,
        data_path: str,
        image_key_path: Optional[str],
        darks: DarksFlatsFileConfig,
        flats: DarksFlatsFileConfig,
        angles: AnglesConfig,
    ):
        self.pattern = Pattern.projection
        self.method_name = "standard_tomo"
        self.package_name = "httomo"
        self._detector_x: int = 0
        self._detector_y: int = 0
        self._angles_total: int = 0
        self.comm = comm
        self.in_file = in_file
        self.data_path = data_path
        self.image_key_path = image_key_path
        self.darks = darks
        self.flats = flats
        self.angles = angles

    def make_data_source(self) -> DataSetSource:
        assert self.pattern in [Pattern.sinogram, Pattern.projection]
        loader = StandardTomoLoader(
            self.in_file,
            self.data_path,
            self.image_key_path,
            self.darks,
            self.flats,
            self.angles,
            1 if self.pattern == Pattern.sinogram else 0,
            self.comm,
        )
        (self._angles_total, self._detector_y, self._detector_x) = loader.global_shape
        return loader

    @property
    def detector_x(self) -> int:
        return self._detector_x

    @property
    def detector_y(self) -> int:
        return self._detector_y

    @property
    def angles_total(self) -> int:
        return self._angles_total


def make_loader(
    repo: MethodRepository, module_path: str, method_name: str, comm: MPI.Comm, **kwargs
) -> LoaderInterface:
    """Produces a loader interface. Only StandardTomoWrapper is supported right now,
    and this method has been added for backwards compatibility. Supporting other loaders
    is a topic that still needs to be explored."""

    if "standard_tomo" not in method_name:
        raise NotImplementedError(
            "Only the standard_tomo loader is currently supported"
        )

    # the following will raise KeyError if not present
    in_file = kwargs["in_file"]
    data_path = kwargs["data_path"]
    image_key_path = kwargs["image_key_path"]
    rotation_angles = kwargs["rotation_angles"]
    angles_path = rotation_angles["data_path"]
    # these will have defaults if not given
    darks: dict = kwargs.get("darks", dict())
    darks_file = darks.get("file", in_file)
    darks_path = darks.get("data_path", data_path)
    darks_image_key = darks.get("image_key_path", image_key_path)
    flats: dict = kwargs.get("darks", dict())
    flats_file = flats.get("file", in_file)
    flats_path = flats.get("data_path", data_path)
    flats_image_key = flats.get("image_key_path", image_key_path)
    # TODO: handle these
    dimension = int(kwargs.get("dimension", 1)) - 1
    preview = kwargs.get("preview", (None, None, None))
    pad = int(kwargs.get("pad", 0))

    return StandardLoaderWrapper(
        comm,
        in_file=in_file,
        data_path=data_path,
        image_key_path=image_key_path,
        darks=DarksFlatsFileConfig(
            file=darks_file, data_path=darks_path, image_key_path=darks_image_key
        ),
        flats=DarksFlatsFileConfig(
            file=flats_file, data_path=flats_path, image_key_path=flats_image_key
        ),
        angles=RawAngles(data_path=angles_path),
    )
