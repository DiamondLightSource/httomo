"""
Module containing prerun functionality for the methods in the HTTomo pipeline.
This includes updating the dictionaries containing the method info.
"""
from collections.abc import Callable
from inspect import signature
from typing import Any, Dict, List, Tuple

from mpi4py import MPI
from numpy import ndarray

from httomo.common import MethodFunc, RunMethodInfo, PlatformSection
from httomo.utils import get_data_in_data_out

comm = MPI.COMM_WORLD


def prerun_method(
    run_method_info: RunMethodInfo,
    section: PlatformSection,
    misc_params: List[Tuple[List[str], object]],
    current_func: MethodFunc,
    dict_datasets_pipeline: Dict[str, ndarray],
):

    run_method_info.package_name = current_func.module_name.split(".")[0]
    dict_params_method = current_func.parameters
    run_method_info.method_name = current_func.method_func.__name__
    func_wrapper = current_func.wrapper_func

    # extra params unrelated to wrapped packages but related to httomo added
    run_method_info.dict_httomo_params = _check_signature_for_httomo_params(
        func_wrapper, current_func, misc_params
    )
    # check platform section object to decide when numpy array need to be returned
    if (run_method_info.task_idx + 1) == len(section.methods):
        # check if the method is the last method in the section
        run_method_info.return_numpy = True
        run_method_info.dict_httomo_params['return_numpy'] = True
        
    
    # Get the information describing if the method is being run only
    # once, or multiple times with different input datasets
    #
    # Make the input and output datasets always be lists just to be
    # generic, and further down loop through all the datasets that the
    # method should be applied to
    data_in, data_out = get_data_in_data_out(
        run_method_info.method_name, dict_params_method
    )

    # Check if the method function's params require any datasets stored
    # in the `datasets` dict
    dataset_params = _check_method_params_for_datasets(
        dict_params_method, dict_datasets_pipeline
    )
    # Update the relevant parameter values according to the required
    # datasets
    dict_params_method.update(dataset_params)

    # check if the module needs the gpu_id parameter and flag it to add in the
    # wrapper
    gpu_id_par = "gpu_id" in signature(current_func.method_func).parameters
    if gpu_id_par:
        dict_params_method.update({"gpu_id": gpu_id_par})

    run_method_info.dict_params_method = dict_params_method
    run_method_info.data_in = data_in
    run_method_info.data_out = data_out


def _check_signature_for_httomo_params(
    func_wrapper: Callable,
    method_name: str,
    possible_extra_params: List[Tuple[List[str], object]],
) -> Dict:
    """Check if the given method requires any parameters related to HTTomo.

    Parameters
    ----------
    func_wrapper : Callable
        Function of a wrapper whose type signature is to be inspected
    method_name : str
        The name of the method to apply.

    possible_extra_params : List[Tuple[List[str], object]]
        Each tuples contains a parameter name and the associated value that
        should be added if a method requires that parameter (note: multiple
        parameter names can be given in case the parameter isn't consistently
        named across tomopy functions, such as "angles" vs "angles_radians")

    Returns
    -------
    Dict
        A dict with the parameter names and values to be added for the given
        method function
    """
    extra_params = {}
    sig_params = signature(func_wrapper).parameters
    for names, val in possible_extra_params:
        for name in names:
            if name in sig_params:
                extra_params[name] = val

    return extra_params


def _check_method_params_for_datasets(
    dict_params_method: Dict, datasets: Dict[str, ndarray]
) -> Dict:
    """Check a given method function's parameter values to see if any of them
    are a dataset.

    Parameters
    ----------
    dict_params_method : Dict
        The dict of param names and their values for a given method function.

    datasets: Dict[str, ndarray]
        The dict of dataset names and their associated arrays.

    Returns
    -------
    Dict
        A dict containing the parameter names assigned to particular datasets,
        and the associated dataset values. This dict can be used to update the
        method function's params dict to reflect the dataset arrays that the
        method function requires.
    """
    dataset_params = {}
    for name, val in dict_params_method.items():
        # If the value of this parameter is a dataset, then replace the
        # parameter value with the dataset value
        if val in datasets:
            dataset_params[name] = datasets[val]

    return dataset_params
