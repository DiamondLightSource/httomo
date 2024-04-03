import pathlib
from typing import Tuple
import numpy as np
import h5py

import httomo
from httomo.runner.dataset import DataSetBlock
from httomo.utils import xp

__all__ = ["calculate_stats", "save_intermediate_data"]


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
    path: str,
    detector_x: int,
    detector_y: int,
    angles: np.ndarray,
) -> None:
    """Saves intermediate data to a file, including auxiliary"""
    if httomo.globals.CHUNK_INTERMEDIATE:
        chunk_shape = [0, 0, 0]
        chunk_shape[slicing_dim] = 1
        DIMS = [0, 1, 2]
        non_slicing_dims = list(set(DIMS) - set([slicing_dim]))
        for dim in non_slicing_dims:
            chunk_shape[dim] = global_shape[dim]
        chunk_shape = tuple(chunk_shape)
    else:
        chunk_shape = None

    # only create if not already present - otherwise return existing dataset
    dataset = file.require_dataset(
        path,
        global_shape,
        data.dtype,
        exact=True,
        chunks=chunk_shape,
    )
    _save_dataset_data(dataset, data, global_shape, global_index)
    _save_auxiliary_data(file, angles, detector_x, detector_y)


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
    dataset[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = data


def _save_auxiliary_data(
    file: h5py.File, angles: np.ndarray, detector_x: int, detector_y: int
):
    # only save if not there yet
    if "/angles" in file:
        return

    file.create_dataset("/angles", data=angles)
    file_name = pathlib.Path(file.filename).name
    file.create_dataset(file_name, data=[0, 0])
    g1 = file.create_group("data_dims")
    g1.create_dataset("detector_x_y", data=[detector_x, detector_y])
