"""
Module contains pre-processing functions that are needed before the main 
loop over GPU blocks has started. 

For example: Centering, Dezingering for multidata, etc. 
"""
from collections.abc import Callable
from inspect import signature
from typing import Any, Dict, List, Tuple, Optional

from mpi4py import MPI
import numpy as np
from numpy import ndarray

from httomo.data.hdf._utils.reslice import single_sino_reslice
from httomo.utils import (
    Colour,
    Pattern,
    _get_slicing_dim,
    log_exception,
    log_once,
    remove_ansi_escape_sequences,
)

def preprocess_data(
    dict_preprocess: Dict[str, ndarray],
    method_funcs: List[Tuple[List[str], object]],
    dict_datasets_pipeline: Dict[str, Optional[ndarray]],
    comm: MPI.Comm,
) -> Dict:
    """_summary_

    Args:
        dict_preprocess (Dict[str, ndarray]): _description_
        method_funcs (List[Tuple[List[str], object]]): _description_
        dict_datasets_pipeline (Dict[str, Optional[ndarray]]): _description_
        comm (MPI.Comm): _description_

    Returns:
        Dict: _description_
    """
    if dict_preprocess['centering']:
        # Finding the CoR is required before the main loop
        if comm.rank == 0:
            single_sino_start_time = MPI.Wtime()
        
        # get the slice index from the parameters list
        slice_for_cor = method_funcs[dict_preprocess['center_method_indx']].parameters['ind']
        if slice_for_cor == "mid" or slice_for_cor is None:
            # local middle value to the preview index
            slice_for_cor = dict_datasets_pipeline[method_funcs[0].parameters["name"]].shape[1] // 2 - 1
        
        # Gather a single sinogram to the rank 0 process
        sino_slice = single_sino_reslice(dict_datasets_pipeline[method_funcs[0].parameters["name"]],
                                         slice_for_cor)
        
        if comm.rank == 0:
            # normalising the sinogram before calculating the CoR
            if dict_datasets_pipeline["flats"] is not None:
                flats1d = np.float32(np.mean(dict_datasets_pipeline["flats"][:,slice_for_cor,:],0))
            else:
                flats1d = 1.0
            if dict_datasets_pipeline["darks"] is not None:
                darks1d = np.float32(np.mean(dict_datasets_pipeline["darks"][:,slice_for_cor,:], 0))
            else:
                darks1d = 0.0
            denom = flats1d - darks1d
            denom[(np.where(denom == 0.0))] = 1.0 
            sino_slice = sino_slice - darks1d / denom

            import cupy as cp
            mid_slice_gpu = cp.array(sino_slice, dtype=cp.float32)
            from httomolibgpu.recon.rotation import find_center_vo
            cor = find_center_vo(mid_slice_gpu)
            single_sino_elapsed_time = MPI.Wtime() - single_sino_start_time
            single_sino_end_str = f"~~~ Single sino reslice + GPU centering ~~~ took {single_sino_elapsed_time} sec to execute, CoR is {cor}"
            log_once(single_sino_end_str, comm=comm, colour=Colour.BVIOLET)
    
    return 0