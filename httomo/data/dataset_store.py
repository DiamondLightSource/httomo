#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2023 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ecpress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>

# ---------------------------------------------------------------------------

import logging
from os import PathLike
from pathlib import Path
import time
import h5py
from typing import List, Literal, Optional, Tuple, Union
from httomo.data.hdf._utils.reslice import reslice
from httomo.data.padding import extrapolate_after, extrapolate_before
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_backing import DataSetStoreBacking
from httomo.runner.dataset_store_interfaces import (
    DataSetSource,
    ReadableDataSetSink,
)
from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
from numpy.typing import DTypeLike
import weakref

from httomo.utils import log_once, make_3d_shape_from_shape


class DataSetStoreWriter(ReadableDataSetSink):
    """A DataSetSink that can be used to store block-wise data in the current chunk (for the current process).

    It uses memory by default - but if there's a memory allocation error, a temporary h5 file is used
    to back the dataset's memory.

    The `make_reader` method can be used to create a DataSetStoreReader from this writer.
    It is intended to be used after the writer has finished, to read the data blockwise again.
    It will use the same underlying data store (h5 file or memory)"""

    def __init__(
        self,
        slicing_dim: Literal[0, 1, 2],
        comm: MPI.Comm,
        temppath: PathLike,
        store_backing: DataSetStoreBacking = DataSetStoreBacking.RAM,
    ):
        self._slicing_dim = slicing_dim
        self._comm = comm

        self._temppath = temppath
        self._readonly = False
        self._h5file: Optional[h5py.File] = None
        self._h5filename: Optional[Path] = None
        self._store_backing = store_backing

        self._data: Optional[Union[np.ndarray, h5py.Dataset]] = None

        self._global_shape: Optional[Tuple[int, int, int]] = None
        self._chunk_shape: Optional[Tuple[int, int, int]] = None
        self._global_index: Optional[Tuple[int, int, int]] = None

        # make sure finalize is called when this object is garbage-collected
        weakref.finalize(self, weakref.WeakMethod(self.finalize))

    @property
    def is_file_based(self) -> bool:
        return self._store_backing is DataSetStoreBacking.File

    @property
    def filename(self) -> Optional[Path]:
        return self._h5filename

    @property
    def comm(self) -> MPI.Comm:
        return self._comm

    # ??? do we need these properties?
    @property
    def global_shape(self) -> Tuple[int, int, int]:
        assert self._global_shape is not None
        return self._global_shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        assert self._chunk_shape is not None
        return self._chunk_shape

    @property
    def global_index(self) -> Tuple[int, int, int]:
        assert self._global_index is not None
        return self._global_index

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    def write_block(self, block: DataSetBlock):
        if self._readonly:
            raise ValueError("Cannot write after creating a reader")
        block.to_cpu()
        start = max(block.chunk_index_unpadded)
        if self._data is None:
            # if non-slice dims in block are different, update the shapes here
            self._global_shape = block.global_shape
            self._chunk_shape = block.chunk_shape_unpadded
            self._global_index = (
                block.global_index_unpadded[0] - block.chunk_index_unpadded[0],
                block.global_index_unpadded[1] - block.chunk_index_unpadded[1],
                block.global_index_unpadded[2] - block.chunk_index_unpadded[2],
            )
            self._aux_data = block.aux_data
            self._create_new_data(block)
        else:
            assert self._global_shape is not None
            assert self._chunk_shape is not None
            assert self._global_index is not None
            if any(self._global_shape[i] != block.global_shape[i] for i in range(3)):
                raise ValueError(
                    "Attempt to write a block with inconsistent shape to existing data"
                )

            if any(
                self._chunk_shape[i] != block.chunk_shape_unpadded[i] for i in range(3)
            ):
                raise ValueError(
                    "Attempt to write a block with inconsistent shape to existing data"
                )

            if any(
                self._global_index[i]
                != block.global_index_unpadded[i] - block.chunk_index_unpadded[i]
                for i in range(3)
            ):
                raise ValueError(
                    "Attempt to write a block with inconsistent shape to existing data"
                )

        # insert the slice here
        assert self._data is not None  # after the above methods, this must be set
        start_idx = [0, 0, 0]
        start_idx[self._slicing_dim] = start
        if self.is_file_based:
            start_idx[self._slicing_dim] += self._global_index[self._slicing_dim]
        self._data[
            start_idx[0] : start_idx[0] + block.shape_unpadded[0],
            start_idx[1] : start_idx[1] + block.shape_unpadded[1],
            start_idx[2] : start_idx[2] + block.shape_unpadded[2],
        ] = block.data_unpadded

    def _get_global_h5_filename(self) -> PathLike:
        """Creates a temporary h5 file to back the storage (using nanoseconds timestamp
        for uniqueness).
        """
        filename = str(Path(self._temppath) / f"httom_tmpstore_{time.time_ns()}.hdf5")
        filename = self.comm.bcast(filename, root=0)

        self._h5filename = Path(filename)
        return self._h5filename

    def _create_new_data(self, block: DataSetBlock):
        if self._store_backing is DataSetStoreBacking.RAM:
            self._data = self._create_numpy_data(
                unpadded_chunk_shape=block.chunk_shape_unpadded,
                dtype=block.data.dtype,
            )
        else:
            log_once(
                "Chunk does not fit in memory - using a file-based store",
                level=logging.WARNING,
            )
            # we create a full file dataset, i.e. file-based,
            # with the full global shape in it
            self._data = self._create_h5_data(
                self.global_shape,
                block.data.dtype,
                self._get_global_h5_filename(),
                self.comm,
            )

    def _create_numpy_data(
        self, unpadded_chunk_shape: Tuple[int, int, int], dtype: DTypeLike
    ) -> np.ndarray:
        """Convenience method to enable mocking easily"""
        return np.empty(unpadded_chunk_shape, dtype)

    def _create_h5_data(
        self,
        global_shape: Tuple[int, int, int],
        dtype: DTypeLike,
        file: PathLike,
        comm: MPI.Comm,
    ) -> h5py.Dataset:
        """Creates a h5 data file based on the file-like object given.
        The returned data object behaves like a numpy array, so can be used freely within
        a DataSet."""

        self._h5file = h5py.File(file, "w", driver="mpio", comm=comm)

        # set how data should be chunked when saving
        # chunks = list(global_shape)
        # chunks[slicing_dim] = 1
        # TODO: best first measure performance impact
        # probably good chunk is (4, 4, <detector_x>)
        h5data = self._h5file.create_dataset("data", global_shape, dtype)

        return h5data

    def make_reader(
        self,
        new_slicing_dim: Optional[Literal[0, 1, 2]] = None,
        padding: Optional[Tuple[int, int]] = None,
    ) -> DataSetSource:
        """Create a reader from this writer, reading from the same store.
        The optional parameter padding can be used if data should be returned with padding slices,
        given as a tuple of (before, after)"""
        if self._data is None:
            raise ValueError("Cannot make reader when no data has been written yet")
        self._readonly = True
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
        reader = DataSetStoreReader(self, new_slicing_dim, padding=padding)
        # make sure finalize is called when reader object is garbage-collected
        weakref.finalize(reader, weakref.WeakMethod(reader.finalize))
        return reader

    def finalize(self):
        self._data = None
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None


class DataSetStoreReader(DataSetSource):
    """Class to read from a store that has previously been written by DataSetStoreWriter,
    in a block-wise fashion.
    """

    def __init__(
        self,
        source: DataSetStoreWriter,
        slicing_dim: Optional[Literal[0, 1, 2]] = None,
        padding: Optional[Tuple[int, int]] = None,
    ):
        self._comm = source.comm
        self._global_shape = source.global_shape
        self._global_index = source.global_index
        self._chunk_shape = source.chunk_shape
        self._aux_data = source.aux_data
        if source._data is None:
            raise ValueError(
                "Cannot create DataSetStoreReader when no data has been written"
            )

        self._h5file: Optional[h5py.File] = None
        self._h5filename: Optional[Path] = None
        source_data = source._data
        if source.is_file_based:
            self._h5filename = source.filename
            self._h5file = h5py.File(source.filename, "r")
            source_data = self._h5file["data"]

        if slicing_dim is None or slicing_dim == source.slicing_dim:
            self._slicing_dim = source.slicing_dim
            self._data = source_data
        else:
            self._data = self._reslice(source.slicing_dim, slicing_dim, source_data)
            self._slicing_dim = slicing_dim

        self._padding = (0, 0) if padding is None else padding
        if self._padding != (0, 0):
            # correct indices for padding
            global_index_t = list(self._global_index)
            global_index_t[self.slicing_dim] -= self._padding[0]
            self._global_index = make_3d_shape_from_shape(global_index_t)

            chunk_shape_t = list(self._chunk_shape)
            chunk_shape_t[self.slicing_dim] += self._padding[0] + self._padding[1]
            self._chunk_shape = make_3d_shape_from_shape(chunk_shape_t)

            if not source.is_file_based:
                self._exchange_neighbourhoods()

        source.finalize()

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def is_file_based(self) -> bool:
        return self._h5filename is not None

    @property
    def filename(self) -> Optional[Path]:
        return self._h5filename

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._global_shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    @property
    def global_index(self) -> Tuple[int, int, int]:
        return self._global_index

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    def _reslice(
        self,
        old_slicing_dim: Literal[0, 1, 2],
        new_slicing_dim: Literal[0, 1, 2],
        data: Union[np.ndarray, h5py.Dataset],
    ) -> Union[np.ndarray, h5py.Dataset]:
        assert old_slicing_dim != new_slicing_dim
        if self._comm.size == 1:
            assert data.shape == self._global_shape
            assert self._comm.size == 1
            return data
        else:
            assert self._comm.size > 1
            if isinstance(data, np.ndarray):  # in-memory chunks
                # we only see a chunk, so we need to do MPI-based reslicing
                array, newdim, startidx = reslice(
                    data, old_slicing_dim + 1, new_slicing_dim + 1, self._comm
                )
                self._chunk_shape = array.shape  #  type: ignore
                assert newdim == new_slicing_dim + 1
                idx = [0, 0, 0]
                idx[new_slicing_dim] = startidx
                self._global_index = (idx[0], idx[1], idx[2])
                return array
            else:
                # we have a full, file-based dataset - all we have to do
                # is calculate the new chunk shape and start index
                rank = self._comm.rank
                nproc = self._comm.size
                length = self._global_shape[new_slicing_dim]
                startidx = round((length / nproc) * rank)
                stopidx = round((length / nproc) * (rank + 1))
                chunk_shape = list(self._global_shape)
                chunk_shape[new_slicing_dim] = stopidx - startidx
                self._chunk_shape = make_3d_shape_from_shape(chunk_shape)
                idx = [0, 0, 0]
                idx[new_slicing_dim] = startidx
                self._global_index = (idx[0], idx[1], idx[2])
                return data

    def _read_block_file(
        self, shape: List[int], dim: int, start_idx: List[int]
    ) -> np.ndarray:
        start_idx[dim] += self._global_index[dim]  # includes padding
        block_data = np.empty(shape, dtype=self._data.dtype)
        before_cut = 0
        after_cut = 0
        # check before boundary
        if start_idx[dim] < 0:
            extrapolate_before(self._data, block_data, -start_idx[dim], dim)
            before_cut = -start_idx[dim]
        # check after boundary
        if start_idx[dim] + shape[dim] > self._data.shape[dim]:
            extrapolate_after(
                self._data,
                block_data,
                start_idx[dim] + shape[dim] - self._data.shape[dim],
                dim,
            )
            after_cut = start_idx[dim] + shape[dim] - self._data.shape[dim]
        slices_read = [slice(None), slice(None), slice(None)]
        slices_read[dim] = slice(
            start_idx[dim] + before_cut, start_idx[dim] + shape[dim] - after_cut
        )
        slices_wrt = [slice(None), slice(None), slice(None)]
        slices_wrt[dim] = slice(before_cut, shape[dim] - after_cut)
        block_data[slices_wrt[0], slices_wrt[1], slices_wrt[2]] = self._data[
            slices_read[0],
            slices_read[1],
            slices_read[2],
        ]
        return block_data

    def _mpi_exchange_padding_area_before(self):
        if self._padding[0] == 0:
            return
        MPI_TAG = 33
        mpi_dtype = dtlib.from_numpy_dtype(self._data.dtype)

        # sender code to right neighbour
        if self._comm.rank < self._comm.size - 1:
            send_slices = [slice(None), slice(None), slice(None)]
            send_slices[self._slicing_dim] = slice(
                self._data.shape[self._slicing_dim]
                - self._padding[0]
                - self._padding[1],
                self._data.shape[self._slicing_dim] - self._padding[1],
            )
            to_send_right_neighbour = np.ascontiguousarray(
                self._data[send_slices[0], send_slices[1], send_slices[2]]
            )
            self._comm.Send(
                [to_send_right_neighbour, mpi_dtype],
                dest=self._comm.rank + 1,
                tag=MPI_TAG,
            )

        # receiver code from right neighbour
        if self._comm.rank > 0:
            recv_shape = list(self._data.shape)
            recv_shape[self._slicing_dim] = self._padding[0]
            receive_buf_from_left_neighbour = np.empty(
                tuple(recv_shape), self._data.dtype
            )
            self._comm.Recv(
                [receive_buf_from_left_neighbour, mpi_dtype],
                source=self._comm.rank - 1,
                tag=MPI_TAG,
            )
            pad_slices = [slice(None), slice(None), slice(None)]
            pad_slices[self._slicing_dim] = slice(0, self._padding[0])

            self._data[pad_slices[0], pad_slices[1], pad_slices[2]] = (
                receive_buf_from_left_neighbour
            )

    def _mpi_exchange_padding_area_after(self):
        MPI_TAG = 44
        if self._padding[1] == 0:
            return
        mpi_dtype = dtlib.from_numpy_dtype(self._data.dtype)

        # sender code to left neighbour
        if self._comm.rank > 0:
            send_slices = [slice(None), slice(None), slice(None)]
            send_slices[self._slicing_dim] = slice(
                self._padding[0], self._padding[0] + self._padding[1]
            )
            to_send_left_neighbour = np.ascontiguousarray(
                self._data[send_slices[0], send_slices[1], send_slices[2]]
            )
            self._comm.Send(
                [to_send_left_neighbour, mpi_dtype],
                dest=self._comm.rank - 1,
                tag=MPI_TAG,
            )

        # receiver code from right neighbour
        if self._comm.rank < self._comm.size - 1:
            recv_shape = list(self._data.shape)
            recv_shape[self._slicing_dim] = self._padding[1]
            receive_buf_from_right_neighbour = np.empty(
                tuple(recv_shape), dtype=self._data.dtype
            )
            self._comm.Recv(
                [receive_buf_from_right_neighbour, mpi_dtype],
                source=self._comm.rank + 1,
                tag=MPI_TAG,
            )
            pad_slices = [slice(None), slice(None), slice(None)]
            pad_slices[self._slicing_dim] = slice(
                self._chunk_shape[self._slicing_dim] - self._padding[1],
                self._chunk_shape[self._slicing_dim],
            )
            self._data[pad_slices[0], pad_slices[1], pad_slices[2]] = (
                receive_buf_from_right_neighbour
            )

    def _extend_data_for_padding(self, core_data: np.ndarray) -> np.ndarray:
        padded_data = np.empty(self._chunk_shape, self._data.dtype)
        core_slices = [slice(None), slice(None), slice(None)]
        core_slices[self._slicing_dim] = slice(
            self._padding[0], self._chunk_shape[self._slicing_dim] - self._padding[1]
        )
        padded_data[core_slices[0], core_slices[1], core_slices[2]] = core_data
        return padded_data

    def _exchange_neighbourhoods(self):
        # we have the core of the chunk in RAM, but without the padding are
        # so we construct the full area with padding in RAM and exchange with MPI

        self._data = self._extend_data_for_padding(self._data)

        # before
        self._mpi_exchange_padding_area_before()
        if self._comm.rank == 0:
            extrapolate_before(
                self._data,
                self._data,
                self._padding[0],
                self._slicing_dim,
                offset=self._padding[0],
            )

        # after
        self._mpi_exchange_padding_area_after()
        if self._comm.rank == self._comm.size - 1:
            extrapolate_after(
                self._data,
                self._data,
                self._padding[1],
                self._slicing_dim,
                offset=self._padding[1],
            )

    def _read_block_ram(
        self, shape: List[int], dim: int, start_idx: List[int]
    ) -> np.ndarray:
        read_slices = [
            slice(start_idx[0], start_idx[0] + shape[0]),
            slice(start_idx[1], start_idx[1] + shape[1]),
            slice(start_idx[2], start_idx[2] + shape[2]),
        ]
        return self._data[read_slices[0], read_slices[1], read_slices[2]]

    def read_block(self, start: int, length: int) -> DataSetBlock:
        shape = list(self._global_shape)
        shape[self._slicing_dim] = length + self._padding[0] + self._padding[1]
        dim = self._slicing_dim

        start_idx = [0, 0, 0]
        start_idx[dim] = start
        if self.is_file_based:
            block_data = self._read_block_file(shape, dim, start_idx)
        else:
            block_data = self._read_block_ram(shape, dim, start_idx)

        return DataSetBlock(
            data=block_data,
            aux_data=self._aux_data,
            slicing_dim=self._slicing_dim,
            block_start=start - self._padding[0],
            chunk_start=self._global_index[self._slicing_dim],
            global_shape=self._global_shape,
            chunk_shape=self._chunk_shape,
            padding=self._padding,
        )

    def finalize(self):
        self._data = None
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
        # also delete the file
        if self._h5filename is not None and self._comm.rank == 0:
            self._h5filename.unlink()
            self._h5filename = None
