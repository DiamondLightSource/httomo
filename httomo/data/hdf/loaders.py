from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional

from h5py import File
from mpi4py.MPI import Comm
from numpy import asarray, deg2rad, ndarray, arange, linspace

from httomo.data.hdf._utils import load
from httomo.utils import _parse_preview, print_once, print_rank

@dataclass
class LoaderData:
    data: ndarray
    flats: ndarray
    darks: ndarray
    angles: ndarray
    angles_total: int
    detector_x: int
    detector_y: int


def standard_tomo(
    name: str,
    in_file: os.PathLike | str,
    data_path: str,
    dimension: int,
    preview: List[Dict[str, int]],
    pad: int,
    comm: Comm,
    image_key_path: Optional[str] = None,
    rotation_angles: Dict[str, Any] = {"data_path": "/entry1/tomo_entry/data/rotation_angle"},
    darks: Optional[Dict] = None,
    flats: Optional[Dict] = None,
) -> LoaderData:
    """Loader for standard tomography data.

    Parameters
    ----------
    name : str
        The name to label the given dataset.
    in_file : Path
        The absolute filepath to the input data.
    data_path : str
        The path within the hdf/nxs file to the data.
    dimension : int
        The dimension to slice in.
    preview : List[Dict[str, int]]
        The previewing/slicing to be applied to the data.
    pad : int
        The padding size to use.
    comm : Comm
        The MPI communicator to use.
    image_key_path : optional, str
        The path within the hdf/nxs file to the image key data.
    rotation_angles : optional, Dict
        A dict that can contain either
        - The path within the hdf/nxs file to the angles data
        - Start, stop, and the total number of angles info to generate a list of
          angles
    darks : optional, Dict
        A dict containing filepath and dataset information about the darks if
        they are not in the same dataset as the data.
    flats : optional, Dict
        A dict containing filepath and dataset information about the flats if
        they are not in the same dataset as the data.

    Returns
    -------
    LoaderData
        The values that all loader functions return.
    """
    with File(in_file, "r", driver="mpio", comm=comm) as file:
        dataset = file[data_path]
        shape = dataset.shape

    if comm.rank == 0:
        print("\033[33m" + f"The full dataset shape is {shape}" + "\033[0m")

    # Get indices in data which contain projections
    if image_key_path is not None:
        data_indices = load.get_data_indices(
            str(in_file),
            image_key_path=image_key_path,
            comm=comm,
        )
    else:
        # Assume that only projection data is in `in_file` (no darks/flats), so
        # the "data indices" are simply all images in `in_file`
        data_indices = list(arange(shape[0]))

    # Get the angles associated to the projection data
    if "data_path" in rotation_angles.keys():
        angles_degrees = load.get_angles(
            str(in_file), path=rotation_angles["data_path"], comm=comm
        )
    else:
        angles_info = rotation_angles["user_defined"]
        angles_degrees = linspace(
            angles_info["start_angle"],
            angles_info["stop_angle"],
            angles_info["angles_total"],
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
        str(in_file), dim, data_path, preview=preview_str, pad=pad_values, comm=comm
    )

    # Get darks and flats
    if darks is not None and flats is not None and darks["file"] != flats["file"]:
        # Get darks and flats from different datasets within different NeXuS
        # files
        darks_data = load.get_darks_flats_separate(
            darks["file"], darks["data_path"], dim=dimension, preview=preview_str
        )
        flats_data = load.get_darks_flats_separate(
            flats["file"], flats["data_path"], dim=dimension, preview=preview_str
        )
    elif darks is not None and flats is not None and darks["file"] == flats["file"]:
        # Get darks and flats from different datasets within the same NeXuS file
        darks_data, flats_data = load.get_darks_flats_together(
            str(in_file),
            data_path,
            darks_path=darks["data_path"],
            flats_path=flats["data_path"],
            image_key_path=image_key_path,
            comm=comm,
            preview=preview_str,
            dim=dimension,
        )
    else:
        # Get darks and flats from the same dataset within the same NeXuS file
        darks_data, flats_data = load.get_darks_flats_together(
            str(in_file),
            data_path,
            image_key_path=image_key_path,
            comm=comm,
            preview=preview_str,
            dim=dimension,
        )

    (angles_total, detector_y, detector_x) = data.shape
    print_rank(
        f"Data shape is {(angles_total, detector_y, detector_x)}"
        + f" of type {data.dtype}",
        comm,
    )

    return LoaderData(data=data, 
                      flats=asarray(flats_data), 
                      darks=asarray(darks_data), 
                      angles=angles, 
                      angles_total=angles_total, 
                      detector_y=detector_y, 
                      detector_x=detector_x)
