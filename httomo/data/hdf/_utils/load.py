import math
from typing import Dict, List, Tuple, Optional, Union

import h5py as h5
import numpy as np
from mpi4py import MPI
from numpy import ndarray

from httomo.utils import log_once


def load_data(
    file: str,
    dim: int,
    path: str,
    preview: str = ":,:,:",
    pad: Tuple = (0, 0),
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Tuple[ndarray, int]:
    """Load data in parallel, slicing it through a certain dimension.

    Parameters
    ----------
    file : str
        Path to file containing the dataset.
    dim : int
        Dimension along which data is sliced when being split between MPI
        processes.
    path : str
        Path to dataset within the file.
    preview : str
        Crop the data with a preview:
    pad : Tuple
        Pad the data by this number of slices.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[ndarray, int]
        The numpy array that has been loaded and the start index in the global shape.
    """
    log_once(f"Loading data: {file}", level=1)
    log_once(f"Path to data: {path}", level=1)
    log_once(f"Preview: ({preview})", level=1)
    if dim == 1:
        return read_through_dim1(file, path, preview=preview, pad=pad, comm=comm)
    elif dim == 2:
        return read_through_dim2(file, path, preview=preview, pad=pad, comm=comm)
    elif dim == 3:
        return read_through_dim3(file, path, preview=preview, pad=pad, comm=comm)
    else:
        raise Exception("Invalid dimension. Choose 1, 2 or 3.")


def read_through_dim3(
    file: str,
    path: str,
    preview: str = ":,:,:",
    pad: Tuple = (0, 0),
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Tuple[ndarray, int]:
    """Read a dataset in parallel, with each MPI process loading a block.

    Parameters
    ----------
    file : str
        Path to file containing the dataset.
    path : str
        Path to dataset within the file.
    preview : str
        Crop the data with a preview:
    pad : Tuple
        Pad the data by this number of slices.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[ndarray, int]
        ADD DESC
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        # Turning the preview into a length and offset. Data will be read from
        # data[offset] to data[offset + length].
        if slice_list[2] == slice(None):
            length = dataset.shape[2]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[2].start is None else slice_list[1].start
            stop = (
                dataset.shape[2] if slice_list[2].stop is None else slice_list[2].stop
            )
            step = 1 if slice_list[2].step is None else slice_list[2].step
            length = (
                stop - start
            ) // step  # Total length of the section of the dataset being read.
            offset = start  # Offset where the dataset will start being read.
        # Bounds of the data this process will load. Length is split between number of
        # processes.
        chunk_idx = round((length / nproc) * rank)
        next_chunk_idx = round((length / nproc) * (rank + 1))
        i0 = chunk_idx + offset - pad[0]
        i1 = next_chunk_idx + offset + pad[1]
        # Checking that i0 and i1 are still within the bounds of the dataset after
        # padding.
        if i0 < 0:
            i0 = 0
        if i1 > dataset.shape[2]:
            i1 = dataset.shape[2]
        data = dataset[:, :, i0:i1:step]
        return data, chunk_idx


def read_through_dim2(
    file: str,
    path: str,
    preview: str = ":,:,:",
    pad: Tuple = (0, 0),
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Tuple[ndarray, int]:
    """Read a dataset in parallel, with each MPI process loading a block.

    Parameters
    ----------
    file : str
        Path to file containing the dataset.
    path : str
        Path to dataset within the file.
    preview : str
        Crop the data with a preview:
    pad : Tuple
        Pad the data by this number of slices.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[ndarray, int]
        The data loaded and its starting index in the global shape
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        # Turning the preview into a length and offset. Data will be read from
        # data[offset] to data[offset + length].
        if slice_list[1] == slice(None):
            length = dataset.shape[1]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[1].start is None else slice_list[1].start
            stop = (
                dataset.shape[1] if slice_list[1].stop is None else slice_list[1].stop
            )
            step = 1 if slice_list[1].step is None else slice_list[1].step
            length = (
                stop - start
            ) // step  # Total length of the section of the dataset being read.
            offset = start  # Offset where the dataset will start being read.
        # Bounds of the data this process will load. Length is split between number of
        # processes.
        chunk_idx = round((length / nproc) * rank)
        next_chunk_idx = round((length / nproc) * (rank + 1))
        i0 = chunk_idx + offset - pad[0]
        i1 = next_chunk_idx + offset + pad[1]
        # Checking that i0 and i1 are still within the bounds of the dataset after
        # padding.
        if i0 < 0:
            i0 = 0
        if i1 > dataset.shape[1]:
            i1 = dataset.shape[1]
        data = dataset[slice_list[0], i0:i1:step, :]
        return data, chunk_idx


def read_through_dim1(
    file: str,
    path: str,
    preview: str = ":,:,:",
    pad: Tuple = (0, 0),
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Tuple[ndarray, int]:
    """Read a dataset in parallel, with each MPI process loading a block.

    Parameters
    ----------
    file : str
        Path to file containing the dataset.
    path : str
        Path to dataset within the file.
    preview : str
        Crop the data with a preview:
    pad : Tuple
        Pad the data by this number of slices.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[ndarray, int]
        The data loaded and its starting index in the global shape
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        # Turning the preview into a length and offset. Data will be read from
        # data[offset] to data[offset + length].
        if slice_list[0] == slice(None):
            length = dataset.shape[0]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[0].start is None else slice_list[0].start
            stop = (
                dataset.shape[0] if slice_list[0].stop is None else slice_list[0].stop
            )
            step = 1 if slice_list[0].step is None else slice_list[0].step
            length = (
                stop - start
            ) // step  # Total length of the section of the dataset being read.
            offset = start  # Offset where the dataset will start being read.
        # Bounds of the data this process will load. Length is split between number of
        # processes.
        chunk_idx = round((length / nproc) * rank)
        next_chunk_idx = round((length / nproc) * (rank + 1))
        i0 = chunk_idx + offset - pad[0]
        i1 = next_chunk_idx + offset + pad[1]
        # Checking that i0 and i1 are still within the bounds of the dataset after
        # padding.
        if i0 < 0:
            i0 = 0
        if i1 > dataset.shape[0]:
            i1 = dataset.shape[0]
        data = dataset[i0:i1:step, slice_list[1], slice_list[2]]
        return data, chunk_idx


def get_pad_values(
    pad: int,
    dim: int,
    dim_length: int,
    data_indices: List[int] = None,
    preview: str = ":,:,:",
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Tuple[int, int]:
    """Get number of slices the block of data is padded either side.

    Parameters
    ----------
    pad : int
        Number of slices to pad block with.
    dim : int
        Dimension data is to be padded in (same dimension data is sliced in).
    dim_length : int
        Size of dataset in the relevant dimension.
    data_indices : List[int]
        When a dataset has non-data in the dataset (for example darks & flats)
        provide data indices to indicate where in the dataset the data lies.
        Only has an effect when dim = 1, as darks and flats are in projection
        space.
    preview : str
        Preview the data will be cropped by. Should be the same preview given to
        the get_data() method.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[int, int]
        ADD DESC
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    if data_indices is not None and dim == 1:
        bound0 = min(data_indices)
        bound1 = max(data_indices) + 1
    else:
        bound0 = 0
        bound1 = dim_length
    if slice_list[0] == slice(None):
        length = dim_length
        offset = 0
        step = 1
    else:
        start = 0 if slice_list[0].start is None else slice_list[0].start
        stop = dim_length if slice_list[0].stop is None else slice_list[0].stop
        step = 1 if slice_list[0].step is None else slice_list[0].step
        length = (
            stop - start
        ) // step  # Total length of the section of the dataset being read.
        offset = start  # Offset where the dataset will start being read.
    # i0, i1 = range of the data this process will load..
    i0 = round((length / nproc) * rank) + offset - pad
    i1 = round((length / nproc) * (rank + 1)) + offset + pad
    # Checking that after padding, the range is still within the bounds it should be.
    if i0 < bound0:
        pad0 = pad - (bound0 - i0)
    else:
        pad0 = pad
    if i1 > bound1:
        pad1 = pad - (i1 - bound1)
    else:
        pad1 = pad
    return pad0, pad1


def get_num_chunks(filepath: str, path: str, comm: MPI.Comm) -> int:
    """Gets the number of chunks in a file.

    Parameters
    ----------
    filepath : str
        The hdf5 file to read from.
    path : str
        The key of the dataset within the file.
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    int
        ADD DESC
    """
    with h5.File(filepath, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        shape = dataset.shape
        chunks = dataset.chunks

    chunk_boundaries = [
        [None] * (math.ceil(shape[i] / chunks[i]) + 1) for i in range(len(shape))
    ]

    # Creating a list of chunk boundaries in each dimension.
    for dim in range(len(shape)):
        boundary = 0
        for i in range(len(chunk_boundaries[dim])):
            if boundary > shape[dim]:
                boundary = shape[dim]
            chunk_boundaries[dim][i] = boundary
            boundary += chunks[dim]

    # Calculating number of chunks
    nchunks = 1
    for dim in range(len(chunk_boundaries)):
        nchunks *= len(chunk_boundaries[dim]) - 1

    return nchunks


def get_angles(file: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD) -> ndarray:
    """Get angles.

    Parameters
    ----------
    file : str
        Path to file containing the data and angles.
    path : str
        Path to the angles within the file.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    ndarray
        A numpy array containing the angles within the give dataset.
    """
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        angles = file[path][...]
    return angles


def _parse_ignore_darks_flats(ignore: Dict) -> List[int]:
    """Get all indices of darks/flats to ignore.

    Parameters
    ----------
    ignore : Dict
        A dict describing the individual and batch darks/flats to ignore.

    Returns
    -------
    List[int]
        A list of indices that describe which darks/flats to ignore.
    """
    indices = []
    if "individual" in ignore.keys():
        for val in ignore["individual"]:
            if not isinstance(val, int):
                err_str = f"The value {val} is not an integer."
                raise ValueError(err_str)
        indices += ignore["individual"]

    if "batch" in ignore.keys():
        for batch in ignore["batch"]:
            if not isinstance(batch["start"], int) or not isinstance(
                batch["stop"], int
            ):
                err_str = (
                    f"The start={batch['start']} and stop={batch['stop']} "
                    f"values should both be integers"
                )
                raise ValueError(err_str)
            indices += list(range(batch["start"], batch["stop"] + 1))
    return indices


def get_darks_flats_together(
    file: str,
    data_path: str = "/entry1/tomo_entry/data/data",
    darks_path: str = None,
    flats_path: str = None,
    image_key_path: str = "/entry1/instrument/image_key/image_key",
    ignore_darks: Optional[Union[bool, Dict]] = False,
    ignore_flats: Optional[Union[bool, Dict]] = False,
    dim: int = 1,
    pad: int = 0,
    preview: str = ":,:,:",
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Tuple[ndarray, ndarray]:
    """Get darks and flats from the same NeXuS file.

    Parameters
    ----------
    file : str
        Path to file containing the dataset.
    data_path : str
        Path to the dataset within the file.
    darks_path : optional, str
        Path to the darks dataset within the file.
    flats_path : optional, str
        Path to the flats dataset within the file.
    image_key_path : str
        Path to the image_key within the file.
    ignore_darks : optional, Union[bool, Dict]
        If bool, specifies ignoring all or none of darks. If dict, specifies
        individual and batch darks to ignore.
    ignore_flats : optional, Union[bool, Dict]
        If bool, specifies ignoring all or none of flats. If dict, specifies
        individual and batch flats to ignore.
    dim : int
        Dimension along which data is being split between MPI processes. Only
        affects darks and flats if dim = 2.
    pad : int
        How many slices data is being padded. Only affects darks and flats if
        dim = 2. (not implemented yet)
    preview : str
        Crop the data with a preview:
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Contains the darks and flats arrays.
    """
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        if darks_path is None and flats_path is None:
            # Get darks and flats from the same dataset within the same NeXuS
            # file
            darks_indices = []
            flats_indices = []
            # Collect indices corresponding to darks and flats
            for i, key in enumerate(file[image_key_path]):
                if int(key) == 1:
                    flats_indices.append(i)
                elif int(key) == 2:
                    darks_indices.append(i)
            # Get indices of darks to ignore (if any)
            if ignore_darks is True:
                darks_indices = []
            elif isinstance(ignore_darks, dict):
                ignore_darks_indices = _parse_ignore_darks_flats(ignore_darks)
                if not set(ignore_darks_indices) <= set(darks_indices):
                    err_str = (
                        f"The darks indices to ignore are "
                        f"{ignore_darks_indices}, which has one or "
                        f"more values outside the darks in the dataset."
                    )
                    raise ValueError(err_str)
                darks_indices = list(set(darks_indices) - set(ignore_darks_indices))
            # Get indices of flats to ignore (if any)
            if ignore_flats is True:
                flats_indices = []
            elif isinstance(ignore_flats, dict):
                ignore_flats_indices = _parse_ignore_darks_flats(ignore_flats)
                if not set(ignore_flats_indices) <= set(flats_indices):
                    err_str = (
                        f"The flats indices to ignore are "
                        f"{ignore_flats_indices}, which has one or "
                        f"more values outside the flats in the dataset."
                    )
                    raise ValueError(err_str)
                flats_indices = list(set(flats_indices) - set(ignore_flats_indices))
            dataset = file[data_path]
            darks = _get_darks_flats(dataset, darks_indices, dim, pad, preview, comm)
            flats = _get_darks_flats(dataset, flats_indices, dim, pad, preview, comm)
        else:
            # Get darks and flats from different datasets within the same NeXuS
            # file
            darks_dataset = file[darks_path]
            darks_indices = np.arange(darks_dataset.shape[0])
            darks = _get_darks_flats(
                darks_dataset, darks_indices, dim, pad, preview, comm
            )
            flats_dataset = file[flats_path]
            flats_indices = np.arange(flats_dataset.shape[0])
            flats = _get_darks_flats(
                flats_dataset, flats_indices, dim, pad, preview, comm
            )

    return darks, flats


def get_darks_flats_separate(
    file_path: str,
    data_path: str,
    dim: int = 1,
    pad: int = 0,
    preview: str = ":,:,:",
    comm: MPI.Comm = MPI.COMM_WORLD,
    ignore_indices: Optional[Union[bool, Dict]] = False,
) -> ndarray:
    """Get darks or flats from a separate dataset and/or separate file from the
    projection data.

    Parameters
    ----------
    file_path : str
        Path to file containing the dataset.
    data_path : str
        Path to the dataset within the file.
    dim : int
        Dimension along which data is being split between MPI processes. Only
        affects darks and flats if dim = 2.
    pad : int
        How many slices data is being padded. Only affects darks and flats if
        dim = 2. (not implemented yet)
    preview : str
        Crop the data with a preview:
    comm : MPI.Comm
        MPI communicator object.
    ignore_indices : optional, Union[bool, Dict]
        If bool, specifies ignoring all or none of darks/flats. If dict,
        specifies individual and batch darks/flats to ignore.

    Returns
    -------
    ndarray
        The darks or flats.
    """
    with h5.File(file_path, "r", driver="mpio", comm=comm) as f:
        dataset = f[data_path]
        indices = list(range(dataset.shape[0]))
        if ignore_indices is True:
            indices = []
        elif isinstance(ignore_indices, dict):
            ignore_indices = _parse_ignore_darks_flats(ignore_indices)
            if not set(ignore_indices) <= set(indices):
                err_str = (
                    f"The darks/flats indices to ignore are "
                    f"{ignore_indices}, which has one or more values "
                    f"outside the darks/flats in the dataset."
                )
                raise ValueError(err_str)
            indices = list(set(indices) - set(ignore_indices))
        data = _get_darks_flats(dataset, indices, dim, pad, preview, comm)
    return data


def _get_darks_flats(
    dataset: h5.Dataset,
    indices: List[int],
    dim: int = 1,
    pad: int = 0,
    preview: str = ":,:,:",
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> ndarray:
    """Get darks or flats array from a given dataset.

    Parameters
    ----------
    dataset : h5.Dataset
        The dataset in which the darks/flats are contained.
    indices : List[int]
        A list of ints which describe the indices at which darks/flats are in
        the dataset.
    dim : int
        Dimension along which data is being split between MPI processes. Only
        affects darks and flats if dim = 2.
    pad : int
        How many slices data is being padded. Only affects darks and flats if
        dim = 2. (not implemented yet)
    preview : str
        Crop the data with a preview:
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    ndarray
        The darks or flats.
    """
    slice_list = get_slice_list_from_preview(preview)
    # If `dim=2`, then the images should be split in the detector_y /
    # vertical dimension across MPI processes, which means that the
    # darks/flats should be split along this dimension. Therefore, some
    # slicing based on the rank of the MPI process needs to be done in order
    # for an MPI process to get its correct share of the data.
    if dim == 2:
        rank = comm.rank
        nproc = comm.size
        if slice_list[1] == slice(None):
            length = dataset.shape[1]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[1].start is None else slice_list[1].start
            stop = (
                dataset.shape[1] if slice_list[1].stop is None else slice_list[1].stop
            )
            step = 1 if slice_list[1].step is None else slice_list[1].step
            length = (stop - start) // step
            offset = start
        i0 = round((length / nproc) * rank) + offset
        i1 = round((length / nproc) * (rank + 1)) + offset
        data = [dataset[x][i0:i1:step][slice_list[2]] for x in indices]
    else:
        data = [dataset[x][slice_list[1], slice_list[2]] for x in indices]
    return data


def get_data_indices(
    filepath: str,
    image_key_path: str = "/entry1/instrument/image_key/image_key",
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> List[int]:
    """Get the indices of where the data is in a dataset.

    Parameters
    ----------
    filepath : str
        Path to the file containing the dataset and image key.
    image_key_path : str
        Path to the image key within the file.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    List[int]
        Contains the indices where data is in the dataset (ie, as opposed to
        indices where darks and flats are).
    """
    with h5.File(filepath, "r", driver="mpio", comm=comm) as f:
        data_indices = []
        for i, key in enumerate(f[image_key_path]):
            if int(key) == 0:
                data_indices.append(i)
    return data_indices


def get_slice_list_from_preview(preview: str) -> List[slice]:
    """Generate slice list to crop data from a preview.

    Parameters
    ----------
    preview : str
        Preview in the form 'start: stop: step, start: stop: step'.

    Returns
    -------
    List[slice, slice, slice]
        A list of slice objects that correspond to the given preview notation.
    """
    slice_list = [None] * 3
    preview = preview.split(",")  # Splitting the dimensions
    for dimension, value in enumerate(preview):
        values = value.split(":")  # Splitting the start, stop, step
        new_values = [None if x.strip() == "" else int(x) for x in values]
        if len(values) == 1:
            slice_list[dimension] = slice(new_values[0])
        elif len(values) == 2:
            slice_list[dimension] = slice(new_values[0], new_values[1])
        elif len(values) == 3:
            slice_list[dimension] = slice(new_values[0], new_values[1], new_values[2])
    return slice_list
