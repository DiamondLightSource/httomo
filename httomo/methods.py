import logging
import pathlib
from typing import Tuple, Union
import numpy as np
import h5py
import hdf5plugin
from mpi4py import MPI

import httomo
from httomo import globals
from httomo.runner.dataset import DataSetBlock
from httomo.utils import log_once, xp

__all__ = ["calculate_stats", "save_intermediate_data"]

# save a copy of the original guess_chunk if it needs to be restored
ORIGINAL_GUESS_CHUNK = h5py._hl.filters.guess_chunk

# The bandwidth that saturates the file system (single process).
# This was estimated using a heuristic approach after performing some
# benchmarks on GPFS03 at DLS using a graph of bandwidth vs message
# size. For more detail, see
# https://github.com/DiamondLightSource/httomo/pull/537
SATURATION_BW = 512 * 2**20


def calculate_stats(
    data: np.ndarray,
) -> Tuple[float, float, float, int]:
    """Calculating the statistics of the given array

    Args:
        data: (np.ndarray): a numpy array

    Returns:
        tuple[(float, float, float, int)]: (min, max, sum, total_elements)
    """

    # do this whereever the data is at the moment (GPU/CPU)
    if getattr(data, "device", None) is not None:
        # GPU
        data = xp.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)
    else:
        # CPU
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)

    return (float(data.min()), float(data.max()), float(data.sum()), data.size)


def save_intermediate_data(
    data: np.ndarray,
    global_shape: Tuple[int, int, int],
    global_index: Tuple[int, int, int],
    slicing_dim: int,
    file: h5py.File,
    frames_per_chunk: int,
    minimum_block_length: int,
    path: str,
    detector_x: int,
    detector_y: int,
    angles: np.ndarray,
) -> None:
    """Saves intermediate data to a file, including auxiliary"""

    if isinstance(file, h5py.File):
        _save_auxiliary_data_hdf5(file, angles, detector_x, detector_y)
        dataset = setup_dataset(
            file,
            path,
            data,
            slicing_dim,
            frames_per_chunk,
            global_shape,
            minimum_block_length,
            filetype="hdf5",
        )

    _save_dataset_data(dataset, data, global_shape, global_index)


def setup_dataset(
    file: h5py.File,
    path: str,
    data: np.ndarray,
    slicing_dim: int,
    frames_per_chunk: int,
    global_shape: Tuple[int, int, int],
    minimum_block_length: int,
    filetype: str,
) -> h5py.Dataset:

    if filetype == "hdf5":
        DIMS = [0, 1, 2]
        non_slicing_dims = list(set(DIMS) - set([slicing_dim]))

        if frames_per_chunk == -1:
            # decide the number of frames in a chunk by maximising the
            # number of frames around the saturation bandwidth of the
            # file system
            # starting value
            sz_per_chunk = data.dtype.itemsize
            for dim in non_slicing_dims:
                sz_per_chunk *= global_shape[dim]

            # the bandwidth is not divided by the number of MPI ranks to
            # provide a consistent chunk size for different number of
            # MPI ranks
            frames_per_chunk = SATURATION_BW // sz_per_chunk

        if frames_per_chunk > data.shape[slicing_dim]:
            warn_message = (
                f"frames_per_chunk={frames_per_chunk} exceeds number of elements in "
                f"slicing dim={slicing_dim} of data with shape {data.shape}. Falling "
                "back to 1 frame per-chunk"
            )
            log_once(warn_message, logging.DEBUG)
            frames_per_chunk = 1

        if frames_per_chunk > minimum_block_length:
            warn_message = (
                f"frames_per_chunk={frames_per_chunk} exceeds length of smallest block "
                f"within section ({minimum_block_length}), in slicing_dim={slicing_dim} "
                f"of data with shape {data.shape}. Falling back to {minimum_block_length} "
                "frames per chunk"
            )
            log_once(warn_message, logging.DEBUG)
            frames_per_chunk = minimum_block_length

        if frames_per_chunk > 0:
            chunk_shape = [0, 0, 0]
            chunk_shape[slicing_dim] = frames_per_chunk
            for dim in non_slicing_dims:
                chunk_shape[dim] = global_shape[dim]
            chunk_shape = tuple(chunk_shape)
        else:
            chunk_shape = None
        # monkey-patch guess_chunk in h5py for compression
        # this is to avoid FILL_TIME_ALLOC
        compression: Union[dict, hdf5plugin.Blosc]
        if httomo.globals.COMPRESS_INTERMEDIATE:
            compression = hdf5plugin.Blosc()
            h5py._hl.filters.guess_chunk = lambda *args, **kwargs: None
        else:
            compression = {}
            h5py._hl.filters.guess_chunk = ORIGINAL_GUESS_CHUNK

        # create a dataset creation property list
        if chunk_shape is not None:
            dcpl = _dcpl_fill_never(chunk_shape, global_shape)
        else:
            dcpl = None

        # adjust the raw data chunk cache options of the dataset
        # according to the chunk size
        if chunk_shape is not None:
            num_chunks = np.prod(
                np.asarray(global_shape) / np.asarray(chunk_shape)
            ).astype(int)
            rdcc_opts = {
                "rdcc_nbytes": data.dtype.itemsize * np.prod(chunk_shape),
                "rdcc_w0": 1,
                "rdcc_nslots": _get_rdcc_nslots(num_chunks),
            }
        else:
            rdcc_opts = {
                "rdcc_nbytes": None,
                "rdcc_w0": None,
                "rdcc_nslots": None,
            }

        # only create if not already present - otherwise return existing dataset
        dataset = file.require_dataset(
            path,
            global_shape,
            data.dtype,
            exact=True,
            chunks=None,  # set in dcpl
            **compression,
            dcpl=dcpl,
            **rdcc_opts,
        )
    return dataset


def _save_dataset_data(
    dataset: h5py.Dataset,
    data: np.ndarray,
    global_shape: Tuple[int, int, int],
    global_index: Tuple[int, int, int],
):
    start = np.array(global_index)
    stop = start + np.array(data.shape)
    assert getattr(data, "device", None) is None, "data must be on CPU for saving"
    assert stop[0] <= dataset.shape[0]
    assert stop[1] <= dataset.shape[1]
    assert stop[2] <= dataset.shape[2]
    assert dataset.shape == global_shape
    if isinstance(dataset, h5py.Dataset) and httomo.globals.COMPRESS_INTERMEDIATE:
        # Write operations must be collective when applying compression, see
        # https://github.com/h5py/h5py/issues/1564
        with dataset.collective:
            dataset[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = data
        return

    dataset[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = data


def _save_auxiliary_data_hdf5(
    file: h5py.File,
    angles: np.ndarray,
    detector_x: int,
    detector_y: int,
):
    # only save if not there yet
    if "/angles" in file:
        return

    file.create_dataset("angles", data=angles)

    file_name = pathlib.Path(file.filename).name
    file.create_dataset(file_name, data=[0, 0])
    g1 = file.create_group("data_dims")
    g1.create_dataset("detector_x_y", data=[detector_x, detector_y])


def _dcpl_fill_never(
    chunk_shape: Union[Tuple[int, int, int], None],
    shape: Tuple[int, int, int],
) -> h5py.h5p.PropDCID:
    """Create a dcpl with specified chunk shape and never fill value."""
    # validate chunk shape (basically a copy from h5py)
    if isinstance(chunk_shape, int) and not isinstance(chunk_shape, bool):
        chunk_shape = (chunk_shape,)
    if isinstance(chunk_shape, tuple) and any(
        chunk > dim for dim, chunk in zip(shape, chunk_shape) if dim is not None
    ):
        errmsg = (
            "Chunk shape must not be greater than data shape in any "
            f"dimension. {chunk_shape} is not compatible with {shape}."
        )
        raise ValueError(errmsg)

    # dcpl initialisation
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)

    dcpl.set_chunk(chunk_shape)

    # we are not going to resize the dataset
    dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)

    return dcpl


def _get_rdcc_nslots(
    num_chunks: int,
) -> int:
    """Estimate the value of rdcc_nslots."""
    # ideally this is a prime number and about 100 times the number
    # of chunks for maximum performance
    if 0 <= num_chunks < 500:
        return 50021
    elif 500 <= num_chunks < 1000:
        return 100003
    elif 1000 <= num_chunks < 1500:
        return 150001
    elif 1500 <= num_chunks < 2000:
        return 200003
    elif 2000 <= num_chunks < 2500:
        return 250007
    elif 2500 <= num_chunks < 3000:
        return 300007
    else:
        # +1 to try my luck in getting a prime!
        return num_chunks * 100 + 1
