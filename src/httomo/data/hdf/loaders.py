from pathlib import Path
from typing import Tuple, List, Dict

from h5py import File
from mpi4py.MPI import Comm
from numpy import asarray, deg2rad, ndarray

from httomo.data.hdf._utils import load
from httomo.utils import _parse_preview, print_once, print_rank


def standard_tomo(name: str, in_file: Path, data_path: str, image_key_path: str,
                  dimension: int, preview: List[Dict[str, int]], pad: int,
                  comm: Comm
                  ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int,
                             int, int]:
    """Loader for standard tomography data.

    Parameters
    ----------
    name : str
        The name to label the given dataset.
    in_file : Path
        The absolute filepath to the input data.
    data_path : str
        The path within the hdf/nxs file to the data.
    image_key_path : str
        The path within the hdf/nxs file to the image key data.
    dimension : int
        The dimension to slice in.
    preview : List[Dict[str, int]]
        The previewing/slicing to be applied to the data.
    pad : int
        The padding size to use.
    comm : Comm
        The MPI communicator to use.

    Returns
    -------
    Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int, int, int]
        A tuple of 8 values that all loader functions return.
    """
    with File(in_file, "r", driver="mpio", comm=comm) as file:
        dataset = file[data_path]
        shape = dataset.shape

    if comm.rank == 0:
        print('\033[33m' + f"The full dataset shape is {shape}" + '\033[0m')

    angles_degrees = load.get_angles(in_file, comm=comm)
    data_indices = load.get_data_indices(
        in_file,
        image_key_path=image_key_path,
        comm=comm,
    )
    angles = deg2rad(angles_degrees[data_indices])

    # Get string representation of `preview` parameter
    preview_str = _parse_preview(preview, shape, data_indices)

    dim = dimension
    pad_values = load.get_pad_values(
        pad,
        dim,
        shape[dim - 1],
        data_indices=data_indices,
        preview=preview_str,
        comm=comm,
    )
    print_rank(f"Pad values are {pad_values}.", comm)
    data = load.load_data(
        in_file, dim, data_path, preview=preview_str, pad=pad_values, comm=comm
    )

    darks, flats = load.get_darks_flats(
        in_file,
        data_path,
        image_key_path=image_key_path,
        comm=comm,
        preview=preview_str,
        dim=dimension,
    )
    darks = asarray(darks)
    flats = asarray(flats)

    (angles_total, detector_y, detector_x) = data.shape
    print_rank(
        f"Data shape is {(angles_total, detector_y, detector_x)}"
        + f" of type {data.dtype}",
        comm,
    )

    return data, flats, darks, angles, angles_total, detector_y, detector_x