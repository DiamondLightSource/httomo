from typing import Tuple

import h5py as h5
from mpi4py import MPI
from numpy import ndarray


def save_dataset(
    out_folder: str,
    file_name: str,
    data: ndarray,
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
    i0, i1 = get_data_bounds(data, comm, slice_dim - 1)
    if slice_dim == 1:
        dataset[i0:i1] = data[...]
    elif slice_dim == 2:
        dataset[:, i0:i1] = data[...]
    elif slice_dim == 3:
        dataset[:, :, i0:i1] = data[...]


def get_data_bounds(
    data: ndarray,
    comm: MPI.Comm,
    slice_dim: int,
) -> Tuple[int, int]:
    """Calculate the bounds of the data that the current MPI process has with
    respect to the full data size.

    When projections have not been removed, the calcualtion of the bounds is
    simpler and can be done independent in each process based solely on:
    - the length of the dimension that the data is being split along
    - the number of processes

    via the following:
    ```
    rank = comm.rank
    nproc = comm.size
    length = dataset.shape[slice_dim - 1]
    i0 = round((length / nproc) * rank)
    i1 = round((length / nproc) * (rank + 1))
    ```

    However, to be more general and work for the case when projections are
    removed, the following approach is instead used.
    """
    # The list returned is ordered by the rank of the process
    recv_data = comm.allgather(data.shape)
    # Find the start bound for the current MPI process based on the sizes of the
    # data that other MPI processes have
    start = 0
    for i in range(comm.rank):
        start += recv_data[i][slice_dim]
    end = start + data.shape[slice_dim]
    return start, end


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
    shape = list(data.shape)
    lengths = comm.gather(shape[dim], 0)
    lengths = comm.bcast(lengths, 0)
    shape[dim] = sum(lengths)
    shape = tuple(shape)
    return shape
