from typing import Tuple

import h5py as h5
from mpi4py import MPI
from numpy import ndarray
from httomo.data.hdf.loaders import LoaderData


def save_dataset(
    out_folder: str,
    file_name: str,
    data: ndarray,
    angles: ndarray,
    detector_x: int,
    detector_y: int,
    slice_dim: int = 1,
    chunks: Tuple = (150, 150, 10),
    path: str = "/data",
    reslice: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """Save dataset in parallel.

    Parameters
    ----------
    out_folder : str
        Path to output folder.
    file_name : str
        Name of file to save dataset in.
    data : ndarray
        Data to save to file.
    loader_info: Dict
        Dictionary with information about the loaded data.
    slice_dim : int
        Where data has been parallelized (split into blocks, each of which is
        given to an MPI process), provide the dimension along which the data was
        sliced so it can be pieced together again.
    chunks : Tuple
        Specify how the data should be chunked when saved.
    path : str
        Path to dataset within the file.
    reslice : bool
        Whether or not the dataset being saved is for a reslice operation (in
        which case, the dataset is permitted to be overwritten if the file
        exists).
    comm : MPI.Comm
        MPI communicator object.
    """
    shape = get_data_shape(data, slice_dim - 1, comm)
    dtype = data.dtype
    # If reslicing, can overwrite the intermediate file contents by first
    # truncating the file
    file_mode = "w" if reslice else "a"
    with h5.File(
        f"{out_folder}/{file_name}", file_mode, driver="mpio", comm=comm
    ) as file:
        dataset = file.create_dataset(path, shape, dtype, chunks=chunks)
        save_data_parallel(dataset, data, slice_dim)
        file.create_dataset("/angles", data=angles)
        file.create_dataset(file_name, data=[0, 0])
        g1 = file.create_group("data_dims")
        g1.create_dataset("detector_x_y", data=[detector_x, detector_y])


def save_data_parallel(
    dataset: h5.Dataset,
    data: ndarray,
    slice_dim: int,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """Save data to dataset in parallel.

    Parameters
    ----------
    dataset : h5.Dataset
        Dataset to save data to.
    data : ndarray
        Data to save to dataset.
    slice_dim : int
        Where data has been parallelized (split into blocks, each of which is
        given to an MPI process), provide the dimension along which the data was
        sliced so it can be pieced together again.
    comm : MPI.Comm
        MPI communicator object.
    """
    rank = comm.rank
    nproc = comm.size
    length = dataset.shape[slice_dim - 1]
    i0 = round((length / nproc) * rank)
    i1 = round((length / nproc) * (rank + 1))
    if slice_dim == 1:
        dataset[i0:i1] = data[...]
    elif slice_dim == 2:
        dataset[:, i0:i1] = data[...]
    elif slice_dim == 3:
        dataset[:, :, i0:i1] = data[...]


def get_data_shape_and_offset(
    data: ndarray, dim: int, comm: MPI.Comm = MPI.COMM_WORLD
) -> Tuple[Tuple, Tuple]:
    """Gets the shape of the distribtued dataset as well as the offset of the
    local one within the full data shape.

    Parameters
    ----------
    data : ndarray
        The process data.
    dim : int
        The dimension in which to get the shape.
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    Tuple, Tuple
        The shape of the given distributed dataset and the global starting index of this dataset,
        both as a 3-tuple of integers
    """
    shape = list(data.shape)
    lengths = comm.gather(shape[dim], 0)
    lengths = comm.bcast(lengths, 0)
    shape[dim] = sum(lengths)
    global_index = [0, 0, 0]
    global_index[dim] = sum(lengths[: comm.rank - 1])
    global_index = tuple(global_index)
    shape = tuple(shape)
    return shape, global_index


def get_data_shape(data: ndarray, dim: int, comm: MPI.Comm = MPI.COMM_WORLD) -> Tuple:
    """Gets the shape of a distributed dataset.

    Parameters
    ----------
    data : ndarray
        The process data.
    dim : int
        The dimension in which to get the shape.
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    Tuple
        The shape of the given distributed dataset.
    """
    return get_data_shape_and_offset(data, dim, comm)[0]
