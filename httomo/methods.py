import pathlib
from typing import Tuple
import numpy as np
import h5py

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

    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)

    return (np.min(data), np.max(data), np.sum(data), data.size)


def save_intermediate_data(
    block: DataSetBlock, file: h5py.File, path: str, detector_x: int, detector_y: int
) -> None:
    """Saves intermediate data to a file, including auxiliary"""
    # only create if not already present - otherwise return existing dataset
    dataset = file.require_dataset(
        path, block.global_shape, block.data.dtype, exact=True
    )
    _save_dataset_data(dataset, block)
    _save_auxiliary_data(file, block, detector_x, detector_y)


def _save_dataset_data(dataset: h5py.Dataset, block: DataSetBlock):
    start = np.array(block.global_index)
    stop = start + np.array(block.shape)
    data = block.data if block.is_cpu else xp.asnumpy(block.data)
    assert stop[0] <= dataset.shape[0]
    assert stop[1] <= dataset.shape[1]
    assert stop[2] <= dataset.shape[2]
    assert dataset.shape == block.global_shape
    dataset[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = data


def _save_auxiliary_data(
    file: h5py.File, block: DataSetBlock, detector_x: int, detector_y: int
):
    # only save if not there yet
    if "/angles" in file:
        return

    file.create_dataset("/angles", data=block.angles)
    file_name = pathlib.Path(file.filename).name
    file.create_dataset(file_name, data=[0, 0])
    g1 = file.create_group("data_dims")
    g1.create_dataset("detector_x_y", data=[detector_x, detector_y])
