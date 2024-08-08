import os
from typing import Any, Dict, List, Optional, Union

from mpi4py.MPI import Comm

__all__ = [
    "standard_tomo",
]


def standard_tomo(
    name: str,
    in_file: Union[os.PathLike, str],
    data_path: str,
    dimension: int,
    preview: List[Dict[str, int]],
    pad: int,
    comm: Comm,
    image_key_path: Optional[str] = None,
    rotation_angles: Dict[str, Any] = {
        "data_path": "/entry1/tomo_entry/data/rotation_angle"
    },
    darks: Optional[Dict] = None,
    flats: Optional[Dict] = None,
    ignore_darks: Optional[Union[bool, Dict]] = False,
    ignore_flats: Optional[Union[bool, Dict]] = False,
):
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
    ignore_darks : optional, Union[bool, Dict]
        If bool, specifies ignoring all or none of darks. If dict, specifies
        individual and batch darks to ignore.
    ignore_flats : optional, Union[bool, Dict]
        If bool, specifies ignoring all or none of flats. If dict, specifies
        individual and batch flats to ignore.
    """
    # Note: this function is just here to define the interface for the yaml
    # TODO: remove this completely

    ...  # pragma: nocover
