"""
Module contains pre-processing functions that are needed before the main 
loop over GPU blocks has started. 

For example: Centering, Dezingering for multidata, etc. 
"""
from typing import Tuple, Union

from mpi4py import MPI
import numpy as np
from numpy import newaxis

from httomo.common import MethodFunc
from httomo.data.hdf._utils.reslice import single_sino_reslice

def preprocess_data(
    projs: np.ndarray,
    darks: np.ndarray,
    flats: np.ndarray,
    centering_method_func: MethodFunc,
    comm: MPI.Comm,
) -> Union[float, Tuple[float, float, float, float]]:
    """_summary_

    Args:
        projs: np.ndarray: projection data
        darks: np.ndarray: dark-field data
        flats: np.ndarray: flat-field data
        centering_method_func MethodFunc: object representing centering method
        comm (MPI.Comm): communicator object

    Returns:
        Union[float, Tuple[float, float, float, float]]: result from regular
            centering, or 360 centering
    """
    sino_slice, slice_for_cor = get_sino(projs, centering_method_func.parameters['ind'])

    if comm.rank == 0:
        sino_slice = normalise_sino(
            sino_slice,
            flats[:,slice_for_cor,:],
            darks[:,slice_for_cor,:],
        )

        # preparing everything for the wrapper execution
        method_name = centering_method_func.parameters.pop("method_name")
        func_wrapper = centering_method_func.wrapper_func
        del centering_method_func.parameters["data_in"]
        del centering_method_func.parameters["data_out"]
        del centering_method_func.parameters["ind"]
        
        # run the wrapper
        res = func_wrapper(
            method_name,
            centering_method_func.parameters,
            sino_slice,
            return_numpy=False,
        )
    else:
        res = None

    res = comm.bcast(res, root=0)
    return res


def get_sino(
    data: np.ndarray,
    idx: str,
) -> Tuple[np.ndarray, int]:
    if idx == "mid" or idx is None:
        # local middle value to the preview index
        slice_for_cor = data.shape[1] // 2 - 1

    # Gather a single sinogram to the rank 0 process
    sino_slice = single_sino_reslice(data, slice_for_cor)
    return sino_slice, slice_for_cor


def normalise_sino(
    sino: np.ndarray,
    flats: np.ndarray,
    darks: np.ndarray,
) -> np.ndarray:
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
