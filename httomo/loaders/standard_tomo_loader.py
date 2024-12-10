"""
Loader and wrapper class for loading standard tomography data collected at Diamond Light Source
beamlines.
"""

import logging
import weakref
from pathlib import Path
from typing import Literal, Optional, Tuple

import h5py
import numpy as np
from mpi4py import MPI

from httomo.darks_flats import DarksFlatsFileConfig, get_darks_flats
from httomo.data.padding import extrapolate_after, extrapolate_before
from httomo.loaders.types import AnglesConfig, UserDefinedAngles
from httomo.preview import Preview, PreviewConfig
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.runner.loader import LoaderInterface
from httomo.types import generic_array
from httomo.utils import log_once, make_3d_shape_from_shape

from httomo_backends.methods_database.query import Pattern


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
        padding: Tuple[int, int] = (0, 0),
    ) -> None:
        if slicing_dim != 0:
            raise NotImplementedError("Only slicing dim 0 is currently supported")

        self._in_file = in_file
        self._data_path = data_path
        self._image_key_path = image_key_path
        self._angles = angles
        self._slicing_dim: Literal[0, 1, 2] = slicing_dim
        self._comm = comm
        self._padding = padding
        self._h5file = h5py.File(in_file, "r")
        self._data: h5py.Dataset = self._get_data()
        self._preview = Preview(
            preview_config=preview_config,
            dataset=self._data,
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

        self._aux_data = self._setup_aux_data(darks, flats)
        self._log_info()
        weakref.finalize(self, self.finalize)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def flats(self) -> Optional[generic_array]:
        return self._aux_data.get_flats()

    @property
    def darks(self) -> Optional[generic_array]:
        return self._aux_data.get_darks()

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._global_shape

    @property
    def raw_shape(self) -> Tuple[int, int, int]:
        return self._data.shape

    @property
    def global_index(self) -> Tuple[int, int, int]:
        return self._chunk_index

    def _calculate_chunk_index_slicing_dim(
        self,
        rank: int,
        nprocs: int,
    ) -> int:
        """
        Calculate the index of the chunk that is associated with the given MPI process in the
        slicing dimension, not including padding
        """
        return round((len(self._data_indices) / nprocs) * rank)

    # TODO: Assume projection slice dim for now, and therefore assume chunk index element
    # ordering
    def _calculate_chunk_index(
        self,
        chunk_index_slicing_dim: int,
    ) -> Tuple[int, int, int]:
        """
        Calculates index of chunk relative to the previewed data, including padding
        """
        return (chunk_index_slicing_dim - self._padding[0], 0, 0)

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
        """
        Calculate shape of the chunk that is associated with the given MPI process, excluding
        padding
        """
        return (
            next_proc_chunk_index - current_proc_chunk_index,
            self._global_shape[1],
            self._global_shape[2],
        )

    def read_block(self, start: int, length: int) -> DataSetBlock:
        start_idx = [0, 0, 0]
        start_idx[self._slicing_dim] += start + self._chunk_index[self._slicing_dim]
        block_shape = list(self.global_shape)
        block_shape[self._slicing_dim] = length + self._padding[0] + self._padding[1]
        block_data = np.empty(block_shape, dtype=self._data.dtype)

        # Bools that reflect if an extended read is needed on either the lower or upper
        # boundary of the block, in order to fill in the before/after padded areas
        # respectively.
        #
        # Assume that an extended read is needed on both sides (as that is the most common
        # case), and then reset to `False` later if either:
        # - the "before" padded area will be filled in with an extrapolation of the first slice
        # in the core of the block
        # - the "after" padded area will be filled in with an extrapolation of the last slice
        # in the core of the block
        before_extended_read: bool = True
        after_extended_read: bool = True

        # Fill in numpy array with "before" and "after" padded areas needed for block
        if start_idx[self._slicing_dim] < 0:
            extrapolate_before(
                self._data,
                block_data,
                self._padding[0],
                self._slicing_dim,
                preview_config=self._preview.config,
            )
            before_extended_read = False

        if (
            start_idx[self._slicing_dim] + block_shape[self._slicing_dim]
            > self.global_shape[self._slicing_dim]
        ):
            extrapolate_after(
                self._data,
                block_data,
                self._padding[1],
                self._slicing_dim,
                preview_config=self._preview.config,
            )
            after_extended_read = False

        # Define slicing required to read the necessary parts from the h5py dataset (the core
        # part of the block + any extended reads)
        #
        # `self._chunk_index` includes padding, but padding shouldn't be accounted for when
        # reading the h5py dataset for the core part of the block, so the padding needs to be
        # removed
        chunk_index_unpadded = list(self._chunk_index)
        chunk_index_unpadded[self._slicing_dim] += self._padding[0]
        chunk_shape_unpadded = list(self.chunk_shape)
        chunk_shape_unpadded[self._slicing_dim] -= self._padding[0] + self._padding[1]
        slices_read = [
            slice(
                self._data_offset[0] + chunk_index_unpadded[0],
                self._data_offset[0]
                + chunk_index_unpadded[0]
                + chunk_shape_unpadded[0],
            ),
            slice(
                self._data_offset[1] + chunk_index_unpadded[1],
                self._data_offset[1]
                + chunk_index_unpadded[1]
                + chunk_shape_unpadded[1],
            ),
            slice(
                self._data_offset[2] + chunk_index_unpadded[2],
                self._data_offset[2]
                + chunk_index_unpadded[2]
                + chunk_shape_unpadded[2],
            ),
        ]

        # Define shifts needed for reading to account for if the before/after padded areas have
        # been filled in with an extrapolation beforehand (in which case the padded area should
        # not be written into by this slicing)
        #
        # More specifically:
        # - if `before_extended_read is True`, then an extended read is used to fill in
        # the "before" padded area -> shift the read index down by the before-padding value to
        # perform the extended read
        # - if `before_extended_read is False`, then extrapolation is used to fill in the
        # "before" padded are -> don't shift the read index down by the before-padding value
        #
        # Similarly for `after_extended_read`.
        start_read_shift = self._padding[0] if before_extended_read else 0
        stop_read_shift = self._padding[1] if after_extended_read else 0
        slices_read[self._slicing_dim] = slice(
            self._data_offset[self._slicing_dim]
            + chunk_index_unpadded[self._slicing_dim]
            + start
            - start_read_shift,
            self._data_offset[self._slicing_dim]
            + chunk_index_unpadded[self._slicing_dim]
            + start
            + length
            + stop_read_shift,
        )

        # Define the slicing required to write the core part of the block into the newly
        # created numpy array `block_data` that has the padded areas already filled in
        slices_write = [slice(None)] * 3

        # Define shifts needed for writing to account for if the before/after padded areas have
        # been filled in with an extrapolation beforehand (in which case the padded area should
        # not be written into by this slicing)
        #
        # More specifically:
        # - if `before_extended_read is True`, then an extended read is used to fill in the
        # "before" padded area -> can write the data read from the h5py dataset at index 0
        # along the slicing dim (because the read has been extended to get the "before" padded
        # area from the h5py dataset
        # - if `before_extended_read is False`, then extrapolation is used to fill in the
        # "before" padded are -> can't write the data read from the h5py dataset at index 0
        # along the slicing dim (because the extrapolation has filled in the "before" padded
        # area already)
        #
        # Similarly for `after_extended_read_idx`.
        start_write_idx = 0 if before_extended_read else self._padding[0]
        stop_write_idx = (
            block_shape[self._slicing_dim]
            if after_extended_read
            else block_shape[self._slicing_dim] - self._padding[1]
        )
        slices_write[self._slicing_dim] = slice(start_write_idx, stop_write_idx)

        # Fill in numpy array with the core part of block + any padding from extended reads
        block_data[slices_write[0], slices_write[1], slices_write[2]] = self._data[
            slices_read[0], slices_read[1], slices_read[2]
        ]

        padded_chunk_shape_list = list(self._chunk_shape)
        padded_chunk_shape_list[self._slicing_dim] += (
            self._padding[0] + self._padding[1]
        )

        return DataSetBlock(
            data=block_data,
            aux_data=self._aux_data,
            slicing_dim=self._slicing_dim,
            block_start=start - self._padding[0],
            chunk_start=self._chunk_index[self._slicing_dim],
            global_shape=self._global_shape,
            chunk_shape=make_3d_shape_from_shape(padded_chunk_shape_list),
            padding=self._padding,
        )

    def _get_angles(self) -> np.ndarray:
        if isinstance(self._angles, UserDefinedAngles):
            return np.linspace(
                self._angles.start_angle,
                self._angles.stop_angle,
                self._angles.angles_total,
            )
        return self._h5file[self._angles.data_path][self._data_indices]

    def finalize(self):
        self._h5file.close()

    def _get_data(self) -> h5py.Dataset:
        return self._h5file[self._data_path]

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    def _setup_aux_data(
        self,
        darks_config: DarksFlatsFileConfig,
        flats_config: DarksFlatsFileConfig,
    ) -> AuxiliaryData:
        angles_arr = np.deg2rad(self._get_angles())
        darks_arr, flats_arr = get_darks_flats(
            darks_config,
            flats_config,
            self._preview.config,
        )
        return AuxiliaryData(angles=angles_arr, darks=darks_arr, flats=flats_arr)

    def _log_info(self) -> None:
        log_once(
            f"The full dataset shape is {self._data.shape}",
            level=logging.DEBUG,
        )
        log_once(
            f"Loading data: {self._in_file}",
            level=logging.DEBUG,
        )
        log_once(
            f"Path to data: {self._data_path}",
            level=logging.DEBUG,
        )
        log_once(
            (
                "Preview: ("
                f"{self._preview.config.angles.start}:{self._preview.config.angles.stop}, "
                f"{self._preview.config.detector_y.start}:{self._preview.config.detector_y.stop}, "
                f"{self._preview.config.detector_x.start}:{self._preview.config.detector_x.stop}"
                ")"
            ),
            level=logging.DEBUG,
        )
        log_once(
            f"Data shape is {self._global_shape} of type {self._data.dtype}",
            level=logging.DEBUG,
        )


class StandardLoaderWrapper(LoaderInterface):
    """
    Wrapper around `StandardTomoLoader` to provide its functionality as a data source to the
    runner, while also giving the runner an implementor of `LoaderInterface`.
    """

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
        self._preview = preview
        self.comm = comm
        self.in_file = in_file
        self.data_path = data_path
        self.image_key_path = image_key_path
        self.darks = darks
        self.flats = flats
        self.angles = angles

    def make_data_source(self, padding: Tuple[int, int] = (0, 0)) -> DataSetSource:
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
            padding=padding,
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

    @property
    def preview(self) -> PreviewConfig:
        return self._preview

    @preview.setter
    def preview(self, preview: PreviewConfig):
        """In case of the sweep runner we need to re-set the private preview"""
        self._preview = preview
