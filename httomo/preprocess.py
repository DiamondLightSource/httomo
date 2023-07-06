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
from numpy import ndarray, newaxis

from httomo.common import RunMethodInfo
from httomo.postrun import postrun_method
from httomo.prerun import prerun_method

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
    misc_params: List[Tuple[List[str], object]],
    dict_datasets_pipeline: Dict[str, Optional[ndarray]],
    comm: MPI.Comm,
) -> Dict:
    """_summary_

    Args:
        dict_preprocess (Dict[str, ndarray]): a dictionary with parameters relevant to pre-process methods
        method_funcs (List[Tuple[List[str], object]]): list of method functions
        dict_datasets_pipeline (Dict[str, Optional[ndarray]]): dictionary with parameters
        comm (MPI.Comm): comm

    Returns:
        Dict: dictionary with parameters
    """
    if dict_preprocess['centering']:
        # Finding the CoR is required before the main loop
        #if comm.rank == 0:
        #    single_sino_start_time = MPI.Wtime()        
        current_func = method_funcs[dict_preprocess['center_method_indx']]
        # get the slice index from the parameters list
        slice_for_cor = current_func.parameters['ind']
        if slice_for_cor == "mid" or slice_for_cor is None:
            # local middle value to the preview index
            slice_for_cor = dict_datasets_pipeline[method_funcs[0].parameters["name"]].shape[1] // 2 - 1
        
        current_func.parameters['ind'] = 0
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
            data = sino_slice - darks1d / denom                      
            
            data = data[:, newaxis, :] # increase dim     
            
            #single_sino_elapsed_time = MPI.Wtime() - single_sino_start_time
            #single_sino_end_str = f"~~~ Single sino reslice + GPU centering ~~~ took {single_sino_elapsed_time} sec to execute, CoR is {cor}"
            #log_once(single_sino_end_str, comm=comm, colour=Colour.BVIOLET)
    
    if comm.rank == 0:
        # preparing everything for the wrapper execution
        module_path = current_func.module_name
        method_name = current_func.method_func.__name__
        func_wrapper = current_func.wrapper_func
        package_name = current_func.module_name.split(".")[0]
        
        #: create an object that would be passed along to prerun_method,
        #: run_method, and postrun_method
        run_method_info = RunMethodInfo(task_idx=dict_preprocess['center_method_indx'])
        
        #: prerun - before running the method, update the dictionaries
        prerun_method(
            run_method_info,
            False,
            misc_params,
            current_func,
            dict_datasets_pipeline,
        )
        
        # Assign the `data` parameter of the method to the prepared data array
        run_method_info.dict_httomo_params["data"] = data
        
        # remove methods name from the parameters list of a method
        run_method_info.dict_params_method.pop('method_name')
        
        # run the wrapper
        res = func_wrapper(
            method_name,
            run_method_info.dict_params_method,
            **run_method_info.dict_httomo_params,
        )
        
        # Store the output(s) of the method in the appropriate
        # dataset in the `dict_datasets_pipeline` dict
        if isinstance(res, (tuple, list)):
            # The method produced multiple outputs
            for val, dataset in zip(res, run_method_info.data_out):
                dict_datasets_pipeline[dataset] = val
        else:
            # The method produced a single output
            dict_datasets_pipeline[run_method_info.data_out] = res   
            
    return dict_datasets_pipeline