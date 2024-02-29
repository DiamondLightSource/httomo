import weakref
from pathlib import Path
from typing import Literal, Optional, Tuple

import h5py
import numpy as np
from mpi4py import MPI

from httomo.darks_flats import DarksFlatsFileConfig, get_darks_flats
from httomo.loaders.types import AnglesConfig, UserDefinedAngles
from httomo.preview import Preview, PreviewConfig
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.runner.loader import LoaderInterface
from httomo.utils import Pattern, log_once


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
        preview_config: PreviewConfig,
        slicing_dim: Literal[0, 1, 2],
        comm: MPI.Comm,
    ) -> None:
        if slicing_dim != 0:
            raise NotImplementedError("Only slicing dim 0 is currently supported")

        self._in_file = in_file
        self._data_path = data_path
        self._image_key_path = image_key_path
        self._angles = angles
        self._slicing_dim = slicing_dim
        self._comm = comm
        self._h5file = h5py.File(in_file, "r")
        self._preview = Preview(
            preview_config=preview_config,
            dataset=self._h5file[data_path],
            image_key=(
                self._h5file[image_key_path] if image_key_path is not None else None
            ),
        )

        self._data_indices = self._preview.data_indices
        self._global_shape = self._preview.global_shape
        self._data_offset = (
            self._data_indices[0],
            self._preview.config.detector_y.start,
            self._preview.config.detector_x.start,
        )

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

        angles_arr = np.deg2rad(self._get_angles())
        darks_arr, flats_arr = get_darks_flats(darks, flats, preview_config)
        self._aux_data = AuxiliaryData(
            angles=angles_arr,
            darks=darks_arr,
            flats=flats_arr,
        )
        self._data: h5py.Dataset = self._get_data()
        self._log_info()
        weakref.finalize(self, self.finalize)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def flats(self) -> np.ndarray:
        return self._aux_data.get_flats()

    @property
    def darks(self) -> np.ndarray:
        return self._aux_data.get_darks()

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._global_shape

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        return self._chunk_index

    def _calculate_chunk_index_slicing_dim(
        self,
        rank: int,
        nprocs: int,
    ) -> int:
        """
        Calculate the index of the chunk that is associated with the given MPI process in the
        slicing dimension
        """
        return round((len(self._data_indices) / nprocs) * rank)

    # TODO: Assume projection slice dim for now, and therefore assume chunk index element
    # ordering
    def _calculate_chunk_index(
        self,
        chunk_index_slicing_dim: int,
    ) -> Tuple[int, int, int]:
        """
        Calculates index of chunk relative to the previewed data
        """
        return (chunk_index_slicing_dim, 0, 0)

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    # TODO: Assume projection slice dim for now, and therefore assume chunk shape element
    # ordering
    # TODO: Assume no previewing/cropping in angles dimension
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

    def read_block(self, start: int, length: int) -> DataSetBlock:
        slices = [
            slice(
                self._data_offset[0] + self._chunk_index[0],
                self._data_offset[0] + self._chunk_index[0] + self.chunk_shape[0],
            ),
            slice(
                self._data_offset[1] + self._chunk_index[1],
                self._data_offset[1] + self._chunk_index[1] + self.chunk_shape[1],
            ),
            slice(
                self._data_offset[2] + self._chunk_index[2],
                self._data_offset[2] + self._chunk_index[2] + self.chunk_shape[2],
            ),
        ]
        slices[self._slicing_dim] = slice(
            self._data_offset[self._slicing_dim]
            + self._chunk_index[self._slicing_dim]
            + start,
            self._data_offset[self._slicing_dim]
            + self._chunk_index[self._slicing_dim]
            + start
            + length,
        )

        return DataSetBlock(
            data=self._data[slices[0], slices[1], slices[2]],
            aux_data=self._aux_data,
            slicing_dim=self._slicing_dim,
            block_start=start,
            chunk_start=self._chunk_index[self._slicing_dim],
            global_shape=self._global_shape,
            chunk_shape=self._chunk_shape,
        )

    def _get_angles(self) -> np.ndarray:
        if isinstance(self._angles, UserDefinedAngles):
            return np.linspace(
                self._angles.start_angle,
                self._angles.stop_angle,
                self._angles.angles_total,
            )

        return self._h5file[self._angles.data_path][...]

    def finalize(self):
        self._h5file.close()

    def _get_data(self) -> h5py.Dataset:
        return self._h5file[self._data_path]

    def _log_info(self) -> None:
        log_once(
            f"The full dataset shape is {self._data.shape}",
            comm=self._comm,
        )
        log_once(
            f"Loading data: {self._in_file}",
            comm=self._comm,
        )
        log_once(
            f"Path to data: {self._data_path}",
            comm=self._comm,
        )
        log_once(
            (
                "Preview: ("
                f"{self._preview.config.angles.start}:{self._preview.config.angles.stop}, "
                f"{self._preview.config.detector_y.start}:{self._preview.config.detector_y.stop}, "
                f"{self._preview.config.detector_x.start}:{self._preview.config.detector_x.stop}"
                ")"
            ),
            comm=self._comm,
        )
        log_once(
            f"Data shape is {self._global_shape} of type {self._data.dtype}",
            comm=self._comm,
        )


class StandardLoaderWrapper(LoaderInterface):
    def __init__(
        self,
        comm: MPI.Comm,
        # parameters that should be adjustable from YAML
        in_file: Path,
        data_path: str,
        image_key_path: Optional[str],
        darks: DarksFlatsFileConfig,
        flats: DarksFlatsFileConfig,
        angles: AnglesConfig,
        preview: PreviewConfig,
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
        self.preview = preview

    def make_data_source(self) -> DataSetSource:
        assert self.pattern in [Pattern.sinogram, Pattern.projection]
        loader = StandardTomoLoader(
            in_file=self.in_file,
            data_path=self.data_path,
            image_key_path=self.image_key_path,
            darks=self.darks,
            flats=self.flats,
            angles=self.angles,
            preview_config=self.preview,
            slicing_dim=1 if self.pattern == Pattern.sinogram else 0,
            comm=self.comm,
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
