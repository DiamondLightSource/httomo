"""
Module contains pre-processing functions that are needed before the main 
loop over GPU blocks has started. 

For example: Centering, Dezingering for multidata, etc. 
"""
from typing import Any, Tuple, TypeAlias, Union

import numpy as np
from mpi4py import MPI
from numpy import newaxis

from httomo.common import PreProcessInfo
from httomo.data.hdf._utils.reslice import single_sino_reslice
from httomo.methods_database.query import get_method_info


Centering180Result: TypeAlias = float
Centering360Result: TypeAlias = Tuple[float, float, float]


def dezinging(
    data: np.ndarray[Any, np.dtype[np.float32]],
    preproc_info: PreProcessInfo,
) -> np.ndarray[Any, np.dtype[np.float32]]:

    param_filter = ("method_name")
    is_cupyrun = get_method_info(
        preproc_info.module_path,
        preproc_info.method_name,
        "implementation"
    ) == "gpu_cupy"

    return preproc_info.wrapper_func(
        preproc_info.method_name,
        {k:preproc_info.params[k] for k in preproc_info.params.keys() - param_filter},
        data,
        return_numpy=True,
        cupyrun=is_cupyrun
    )


def centering(
    projs: np.ndarray[Any, np.dtype[np.float32]],
    darks: np.ndarray[Any, np.dtype[np.float32]],
    flats: np.ndarray[Any, np.dtype[np.float32]],
    centering_method_info: PreProcessInfo,
    comm: MPI.Comm,
) -> Union[Centering180Result, Centering360Result]:
    """_summary_

    Args:
        projs: np.ndarray: projection data
        darks: np.ndarray: dark-field data
        flats: np.ndarray: flat-field data
        centering_method_info PreProcessInfo: object representing centering method
        comm (MPI.Comm): communicator object

    Returns:
        Union[float, Tuple[float, float, float, float]]: result from regular
            centering, or 360 centering
    """
    sino_slice, slice_for_cor = _get_sino(projs, centering_method_info.params['ind'])

    if comm.rank == 0:
        sino_slice = _normalise_sino(
            sino_slice,
            flats[:,slice_for_cor,:],
            darks[:,slice_for_cor,:],
        )

        param_filter = ("method_name", "data_in", "data_out", "ind")
        is_cupyrun = get_method_info(
            centering_method_info.module_path,
            centering_method_info.method_name,
            "implementation"
        ) == "gpu_cupy"
        
        res = centering_method_info.wrapper_func(
            centering_method_info.method_name,
            {k:centering_method_info.params[k] for k in centering_method_info.params.keys() - param_filter},
            sino_slice,
            return_numpy=False,
            cupyrun=is_cupyrun
        )
    else:
        res = None

    res: Union[Centering180Result, Centering360Result] = comm.bcast(res, root=0)
    return res


def _get_sino(
    data: np.ndarray[Any, np.dtype[np.float32]],
    idx: str,
) -> Tuple[np.ndarray, int]:
    if idx == "mid" or idx is None:
        # local middle value to the preview index
        slice_for_cor = data.shape[1] // 2 - 1

    # Gather a single sinogram to the rank 0 process
    sino_slice = single_sino_reslice(data, slice_for_cor)
    return sino_slice, slice_for_cor


def _normalise_sino(
    sino: np.ndarray[Any, np.dtype[np.float32]],
    flats: np.ndarray[Any, np.dtype[np.float32]],
    darks: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if flats is not None:
        flats1d = np.float32(np.mean(flats, 0))
    else:
        flats1d = 1.0
    if darks is not None:
        darks1d = np.float32(np.mean(darks))
    else:
        darks1d = 0.0
    denom = flats1d - darks1d
    denom[(np.where(denom == 0.0))] = 1.0
    sino -= darks1d / denom
    return sino[:, newaxis, :] # increase dim