from dataclasses import dataclass, field
import dataclasses
import multiprocessing
from pathlib import Path
import time
from typing import Any, List, Dict, Literal, Optional, Tuple, Union
from collections.abc import Callable
from inspect import signature
from importlib import import_module

from numpy import ndarray
import numpy as np
from mpi4py import MPI
from httomo.data.hdf.loaders import LoaderData

import httomo.globals
from httomo.common import remove_ansi_escape_sequences
from httomo.utils import log_once, Pattern, _get_slicing_dim, Colour, log_exception
from httomo.yaml_utils import open_yaml_config
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo.data.hdf._utils.chunk import save_dataset, get_data_shape
from httomo.data.hdf._utils.reslice import reslice, reslice_filebased
from httomo._stats.globals import min_max_mean_std
from httomo.methods_database.query import get_method_info
from httomolib.misc.images import save_to_images


# TODO: Define a list of savers which have no output dataset and so need to
# be treated differently to other methods. Probably should be handled in a
# more robust way than this workaround.
SAVERS_NO_DATA_OUT_PARAM = ["save_to_images"]

reslice_warn_str = (
    f"WARNING: Reslicing is performed more than once in this "
    f"pipeline, is there a need for this?"
)
# Hardcoded string that is used to check if a method is in a reconstruction
# module or not
RECON_MODULE_MATCH = "recon.algorithm"
MAX_SWEEPS = 1

from httomo.wrappers_class import TomoPyWrapper
from httomo.wrappers_class import HttomolibWrapper


@dataclass
class MethodFunc:
    """
    Class holding information about each tomography pipeline method

    Attributes
    ----------
    module_name : str
        Fully qualified name of the module where the method is. E.g. httomolib.prep.normalize
    method_function : Callable
        The actual method callable
    wrapper_function: Optional[Callable]
        The wrapper function to handle the execution. It may be None,
        for example for loaders.
    parameters : Dict[str, Any]
        The method parameters that are specified in the pipeline yaml file.
        They are used as kwargs when the method is called.
    is_loader : bool
        Whether the method is a loader function
    cpu : bool
        Whether CPU execution is supported.
    gpu : bool
        Whether GPU execution is supported.
    reslice_ahead : bool
        Whether a reslice needs to be done due to a pattern change in the pipeline
    """

    module_name: str
    method_function: Callable
    wrapper_function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_loader: bool = False
    pattern: Pattern = Pattern.projection
    cpu: bool = True
    gpu: bool = False
    reslice_ahead: bool = False


@dataclass
class ResliceInfo:
    """
    Class holding information regarding reslicing

    Attributes
    ----------
    count : int
        Counter how many reslices were done so far
    has_warn_printed : bool
        Whether the reslicing warning has been printed
    reslice_dir : Optional[Path]
        The directory to use with file-based reslicing. If None,
        reslicing will be done in-memory.
    """

    count: int
    has_warn_printed: bool
    reslice_dir: Optional[Path] = None


@dataclass
class PlatformSection:
    """
    Data class to represent a section of the pipeline that runs on the same platform.
    That is, all methods contained in this section of the pipeline run either all on CPU
    or all on GPU.

    This is used to iterate through GPU memory in chunks.

    Attributes
    ----------
    gpu : bool
        Whether this section is a GPU section (True) or CPU section (False)
    pattern : Pattern
        To denote the slicing pattern - sinogram, projection
    max_slices : int
        Holds information about how many slices can be fit in one chunk without
        exhausting memory (relevant on GPU only)
    methods : List[MethodFunc]
        List of methods in this section
    """

    gpu: bool
    pattern: Pattern
    max_slices: int
    methods: List[MethodFunc]


def run_tasks(
    in_file: Path,
    yaml_config: Path,
    dimension: int,
    pad: int = 0,
    ncore: int = 1,
    save_all: bool = False,
    reslice_dir: Optional[Path] = None,
) -> None:
    """Run the tomopy pipeline defined in the YAML config file

    Parameters
    ----------
    in_file : Path
        The file to read data from.
    yaml_config : Path
        The file containing the processing pipeline info as YAML.
    dimension : int
        The dimension to slice in.
    pad : int
        The padding size to use. Defaults to 0.
    ncore : int
        The number of the CPU cores per process.
    save_all : bool
        Specifies if intermediate datasets should be saved for all tasks in the
        pipeline.
    reslice_dir : Optional[Path]
        Path where to store the reslice intermediate files, or None if reslicing
        should be done in-memory.
    """
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        # use all available CPU cores if not an MPI run
        ncore = multiprocessing.cpu_count()

    # Define dict to store arrays of the whole pipeline using provided YAML
    dict_datasets_pipeline = _initialise_datasets(yaml_config, SAVERS_NO_DATA_OUT_PARAM)

    # Define list to store dataset stats for each task in the user config YAML
    glob_stats = _initialise_stats(yaml_config)

    # Get a list of the python functions associated to the methods defined in
    # user config YAML
    method_funcs = _get_method_funcs(yaml_config, comm)

    # Define dict of params that are needed by loader functions
    dict_loader_extra_params = {
        "in_file": in_file,
        "dimension": dimension,
        "pad": pad,
        "comm": comm,
    }

    # A counter to track how many reslices occur in the processing pipeline
    reslice_info = ResliceInfo(count=0, has_warn_printed=False, reslice_dir=reslice_dir)

    # Associate patterns to method function objects
    for i, method_func in enumerate(method_funcs):
        method_funcs[i] = _assign_pattern_to_method(method_func)

    method_funcs = _check_if_should_reslice(method_funcs)
    gpu_sections = _determine_gpu_sections(method_funcs)

    # Check pipeline for the number of parameter sweeps present. If more than
    # one is defined, raise an error, due to not supporting multiple parameter
    # sweeps
    params = [m.parameters for m in method_funcs]
    no_of_sweeps = sum(map(_check_params_for_sweep, params))

    if no_of_sweeps > MAX_SWEEPS:
        err_str = (
            f"There are {no_of_sweeps} parameter sweeps in the "
            f"pipeline, but a maximum of {MAX_SWEEPS} is supported."
        )
        raise ValueError(err_str)

    # start MPI timer for rank 0
    if comm.rank == 0:
        start_time = MPI.Wtime()

    #: add to the console and log file, the full path to the user.log file
    log_once(
        f"See the full log file at: {httomo.globals.run_out_dir}/user.log",
        comm,
        colour=Colour.CYAN,
        level=0,
    )

    # Run the methods
    for idx, method_func in enumerate(method_funcs):
        package = method_func.module_name.split(".")[0]
        method_name = method_func.method_function.__name__
        task_no_str = f"Running task {idx+1}"
        task_end_str = task_no_str.replace("Running", "Finished")
        pattern_str = f"(pattern={method_func.pattern.name})"
        log_once(
            f"{task_no_str} {pattern_str}: {method_name}...",
            comm,
            colour=Colour.LIGHT_BLUE,
            level=0,
        )
        start = time.perf_counter_ns()
        if method_func.is_loader:
            method_func.parameters.update(dict_loader_extra_params)

            # Check if a value for the `preview` parameter of the loader has
            # been provided
            if "preview" not in method_func.parameters.keys():
                method_func.parameters["preview"] = [None]

            loader_data = _run_loader(
                method_func.method_function, method_func.parameters
            )

            # Update `dict_datasets_pipeline` dict with the data that has been
            # loaded by the loader
            dict_datasets_pipeline[method_func.parameters["name"]] = loader_data.data
            dict_datasets_pipeline["flats"] = loader_data.flats
            dict_datasets_pipeline["darks"] = loader_data.darks

            # Extra params relevant to httomo that a wrapper function might need
            possible_extra_params = [
                (["darks"], loader_data.darks),
                (["flats"], loader_data.flats),
                (["angles", "angles_radians"], loader_data.angles),
                (["comm"], comm),
                (["out_dir"], httomo.globals.run_out_dir),
                (["reslice_ahead"], False),
            ]
        else:
            # check if the module needs the ncore parameter and add it
            if "ncore" in signature(method_func.method_function).parameters:
                method_func.parameters.update({"ncore": ncore})

            reslice_info, glob_stats[idx] = _run_method(
                idx,
                save_all,
                possible_extra_params,
                method_func,
                method_funcs[idx - 1],
                method_funcs[idx + 1] if idx < len(method_funcs) - 1 else None,
                dict_datasets_pipeline,
                str(httomo.globals.run_out_dir),
                glob_stats[idx],
                comm,
                reslice_info,
            )

        stop = time.perf_counter_ns()
        output_str_list = [
            f"    {task_end_str} {pattern_str}: {method_name} (",
            package,
            f") Took {float(stop-start)*1e-6:.2f}ms",
        ]
        output_colour_list = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
        log_once(output_str_list, comm=comm, colour=output_colour_list)

    # Log the number of reslice operations peformed in the pipeline
    reslice_summary_str = f"Total number of reslices: {reslice_info.count}"
    reslice_summary_colour = Colour.BLUE if reslice_info.count <= 1 else Colour.RED
    log_once(reslice_summary_str, comm=comm, colour=reslice_summary_colour, level=1)

    elapsed_time = 0.0
    if comm.rank == 0:
        elapsed_time = MPI.Wtime() - start_time
        end_str = f"~~~ Pipeline finished ~~~ took {elapsed_time} sec to run!"
        log_once(end_str, comm=comm, colour=Colour.BVIOLET)
        #: remove ansi escape sequences from the log file
        remove_ansi_escape_sequences(f"{httomo.globals.run_out_dir}/user.log")


def _initialise_datasets(
    yaml_config: Path, savers_no_data_out_param: List[str]
) -> Dict[str, Optional[np.ndarray]]:
    """Add keys to dict that will contain all datasets defined in the YAML
    config.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML
    savers_no_data_out_param : List[str]
        A list of savers which have neither `data_out` nor `data_out_multi` as
        their output.

    Returns
    -------
    Dict
        The dict of datasets, whose keys are the names of the datasets, and
        values will eventually be arrays (but initialised to None in this
        function)
    """
    datasets: Dict[str, Optional[np.ndarray]] = {}
    # Define the params related to dataset names in the given function, whether
    # it's a loader or a method function
    loader_dataset_params = ["name"]
    # TODO: For now, make savers such as `save_to_images` a special case where:
    # - `data_in` is defined
    # - but `data_out` is not, since the output of the method is not a dataset
    # but something else, like a directory containing many subdirectories of
    # images.
    # And therefore, don't try to inspect the `data_out` parameter of the method
    # to then try and initialise a dataset from its value, since it won't exist!
    savers_no_data_out_params = ["data_in"]

    yaml_conf = open_yaml_config(yaml_config)
    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        method_name, method_conf = module_conf.popitem()
        if "loaders" in module_name:
            dataset_params = loader_dataset_params
        elif method_name in savers_no_data_out_param:
            dataset_params = savers_no_data_out_params
        else:
            if (
                "data_in_multi" in method_conf.keys()
                and "data_out_multi" in method_conf.keys()
            ):
                method_dataset_params = ["data_in_multi", "data_out_multi"]
            else:
                method_dataset_params = ["data_in", "data_out"]
            dataset_params = method_dataset_params

        # Add dataset param value to the dict of datasets if it doesn't already
        # exist
        for dataset_param in dataset_params:
            # Check if there are multiple input/output datasets to account for
            if type(method_conf[dataset_param]) is list:
                for dataset_name in method_conf[dataset_param]:
                    if dataset_name not in datasets:
                        datasets[dataset_name] = None
            else:
                if method_conf[dataset_param] not in datasets:
                    datasets[method_conf[dataset_param]] = None

    return datasets


# TODO: There's a lot of overlap with the `_initialise_datasets()` function, the
# only difference is that this function doesn't need to inspect the output
# datasets of a method, so perhaps the two functions can be nicely merged?
def _initialise_stats(yaml_config: Path) -> List[Dict[str, List]]:
    """Generate a list of dicts that will hold the stats for the datasets in all
    the methods in the pipeline.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML

    Returns
    -------
    List[Dict]
        A list containing the stats of all datasets of all methods in the
        pipeline.
    """
    stats: List[Dict[str, List]] = []
    # Define the params related to dataset names in the given function, whether
    # it's a loader or a method function
    loader_dataset_param = "name"

    yaml_conf = open_yaml_config(yaml_config)
    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        _, method_conf = module_conf.popitem()

        if "loaders" in module_name:
            dataset_param = loader_dataset_param
        else:
            if "data_in_multi" in method_conf.keys():
                method_dataset_param = "data_in_multi"
            else:
                method_dataset_param = "data_in"
            dataset_param = method_dataset_param

        # Dict to hold the stats for each dataset associated with the method
        method_stats: Dict[str, List] = {}

        # Check if there are multiple input datasets to account for
        if type(method_conf[dataset_param]) is list:
            for dataset_name in method_conf[dataset_param]:
                method_stats[dataset_name] = []
        else:
            method_stats[method_conf[dataset_param]] = []

        stats.append(method_stats)

    return stats


def _get_method_funcs(yaml_config: Path, comm: MPI.Comm) -> List[MethodFunc]:
    """Gather all the python functions needed to run the defined processing
    pipeline.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML
    comm : MPI.Comm
        MPI communicator object.
    Returns
    -------
    List[MethodFunc]
        A list describing each method function with its properties
    """
    method_funcs: List[MethodFunc] = []
    yaml_conf = open_yaml_config(yaml_config)

    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        split_module_name = module_name.split(".")
        method_name, method_conf = module_conf.popitem()

        if split_module_name[0] == "httomo":
            # deal with httomo loaders
            is_loader = "loaders" in module_name
            module = import_module(module_name)
            method_func: Callable = getattr(module, method_name)
            method_funcs.append(
                MethodFunc(
                    module_name=module_name,
                    method_function=method_func,
                    wrapper_function=None,
                    parameters=method_conf,
                    is_loader=is_loader,
                    cpu=True,
                    gpu=False,
                    reslice_ahead=False,
                    pattern=Pattern.all,
                )
            )
        elif split_module_name[0] == "tomopy":
            # initialise the TomoPy wrapper class
            wrapper_init_module = TomoPyWrapper(
                split_module_name[1], split_module_name[2], method_name, comm
            )
            wrapper_func: Callable = getattr(wrapper_init_module.module, method_name)
            wrapper_method = wrapper_init_module.wrapper_method
            method_funcs.append(
                MethodFunc(
                    module_name=module_name,
                    method_function=wrapper_func,
                    wrapper_function=wrapper_method,
                    parameters=method_conf,
                    is_loader=False,
                    cpu=True,
                    gpu=False,
                    reslice_ahead=False,
                    pattern=Pattern.all,
                )
            )
        elif split_module_name[0] == "httomolib":
            # initialise the httomolib wrapper class
            wrapper_init_module = HttomolibWrapper(
                split_module_name[1], split_module_name[2], method_name, comm
            )
            wrapper_func: Callable = getattr(wrapper_init_module.module, method_name)
            wrapper_method = wrapper_init_module.wrapper_method
            method_funcs.append(
                MethodFunc(
                    module_name=module_name,
                    method_function=wrapper_func,
                    wrapper_function=wrapper_method,
                    parameters=method_conf,
                    is_loader=False,
                    cpu=wrapper_func.meta.cpu,
                    gpu=wrapper_func.meta.gpu,
                    reslice_ahead=False,
                    pattern=Pattern.all,
                )
            )
        else:
            err_str = (
                f"An unknown module name was encountered: " f"{split_module_name[0]}"
            )
            log_exception(err_str)
            raise ValueError(err_str)

    return method_funcs


def _run_method(
    task_idx: int,
    save_all: bool,
    misc_params: List[Tuple[List[str], object]],
    current_func: MethodFunc,
    prev_func: MethodFunc,
    next_func: Optional[MethodFunc],
    dict_datasets_pipeline: Dict[str, Optional[ndarray]],
    out_dir: str,
    glob_stats: Dict,
    comm: MPI.Comm,
    reslice_info: ResliceInfo,
) -> Tuple[ResliceInfo, Dict]:
    """Run a method function in the processing pipeline.

    Parameters
    ----------
    task_idx : int
        The index of the current task (zero-based indexing).
    save_all : bool
        Whether to save the result of all methods in the pipeline,
    misc_params : List[Tuple[List[str], object]]
        A list of possible extra params that may be needed by a method.
    current_func : MethodFunc
        Object describing the python function that performs the method.
    prev_func : MethodFunc
        Object describing the python function that performed the previous method in the pipeline.
    next_func: Optional[MethodFunc]
        Object describing the python function that is next in the pipeline,
        unless the current method is the last one.
    dict_datasets_pipeline : Dict[str, ndarray]
        A dict containing all available datasets in the given pipeline.
    out_dir : str
        The path to the output directory of the run.
    glob_stats : Dict
        A dict of the dataset names to store their associated global stats if
        necessary.
    comm : MPI.Comm
        The MPI communicator used for the run.
    reslice_info : ResliceInfo
        Object tracking the reslicing information, such as the number of reslices, if the
        warning has been printed, and the reslice directory if required.

    Returns
    -------
    Tuple[ResliceInfo, Dict]
        Reslicing information and global stats to enable the information to persist across method executions.
    """

    module_path = current_func.module_name
    package_name = current_func.module_name.split(".")[0]
    dict_params_method = current_func.parameters
    method_name = current_func.method_function.__name__
    func_wrapper = current_func.wrapper_function

    save_result = _check_save_result(
        next_func is None,
        module_path,
        save_all,
        dict_params_method.pop("save_result", None),
    )

    # Check if the input dataset should be resliced before the task runs
    should_reslice = current_func.reslice_ahead
    if should_reslice:
        reslice_info.count += 1
        current_slice_dim = _get_slicing_dim(prev_func.pattern)
        next_slice_dim = _get_slicing_dim(current_func.pattern)

    # the GPU wrapper should know if the reslice is needed to convert the result
    # to numpy from cupy array
    reslice_ahead = next_func.reslice_ahead if next_func is not None else False
    # reslice_ahead must be the last item in the list
    misc_params[-1] = (["reslice_ahead"], reslice_ahead)

    if reslice_info.count > 1 and not reslice_info.has_warn_printed:
        log_once(reslice_warn_str, comm=comm, colour=Colour.RED)
        reslice_info.has_warn_printed = True

    # extra params unrelated to wrapped packages but related to httomo added
    dict_httomo_params = _check_signature_for_httomo_params(
        func_wrapper, current_func.method_function, misc_params
    )

    # Get the information describing if the method is being run only
    # once, or multiple times with different input datasets
    #
    # Make the input and output datasets always be lists just to be
    # generic, and further down loop through all the datasets that the
    # method should be applied to
    if (
        "data_in" in dict_params_method.keys()
        and "data_out" in dict_params_method.keys()
    ):
        data_in = [dict_params_method.pop("data_in")]
        data_out = [dict_params_method.pop("data_out")]
    elif (
        "data_in_multi" in dict_params_method.keys()
        and "data_out_multi" in dict_params_method.keys()
    ):
        data_in = dict_params_method.pop("data_in_multi")
        data_out = dict_params_method.pop("data_out_multi")
    else:
        # TODO: This error reporting is possibly better handled by
        # schema validation of the user config YAML
        if method_name not in SAVERS_NO_DATA_OUT_PARAM:
            err_str = "Invalid in/out dataset parameters"
            raise ValueError(err_str)
        else:
            data_in = [dict_params_method.pop("data_in")]
            data_out = [None]

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
    gpu_id_par = "gpu_id" in signature(current_func.method_function).parameters
    if gpu_id_par:
        dict_params_method.update({"gpu_id": gpu_id_par})

    # Check if method type signature requires global statistics
    req_glob_stats = "glob_stats" in signature(current_func.method_function).parameters

    # Check if a parameter sweep is defined for any of the method's
    # parameters
    current_param_sweep = False
    for k, v in dict_params_method.items():
        if type(v) is tuple:
            param_sweep_name = k
            param_sweep_vals = v
            current_param_sweep = True
            break

    for in_dataset_idx, in_dataset in enumerate(data_in):
        # First, setup the datasets and arrays needed for the method, based on
        # two factors:
        # - if the current method has a parameter sweep for one of its
        #   parameters
        # - if the current method is going to run on the output of a parameter
        #   sweep from a previous method in the pipeline
        if method_name in SAVERS_NO_DATA_OUT_PARAM:
            if current_param_sweep:
                err_str = f"Parameters sweeps on savers is not supported"
                raise ValueError(err_str)
            else:
                if type(dict_datasets_pipeline[in_dataset]) is list:
                    arrs = dict_datasets_pipeline[in_dataset]
                else:
                    arrs = [dict_datasets_pipeline[in_dataset]]
        else:
            if current_param_sweep:
                arrs = [dict_datasets_pipeline[in_dataset]]
            else:
                # If the data is a list of arrays, then it was the result of a
                # parameter sweep from a previous method, so the next method
                # must be applied to all arrays in the list
                if type(dict_datasets_pipeline[in_dataset]) is list:
                    arrs = dict_datasets_pipeline[in_dataset]
                else:
                    arrs = [dict_datasets_pipeline[in_dataset]]

        # Both `data_in` and `data_out` are lists. However, `data_out` for a
        # method can be such that there are multiple output datasets produced by
        # the method when given only one input dataset.
        #
        # In this case, there will be a list of dataset names within `data_out`
        # at some index `j`, which is then associated with the single input
        # dataset in `data_in` at index `j`.
        #
        # Therefore, set the output dataset to be the element in `data_out` that
        # is at the same index as the input dataset in `data_in`
        out_dataset = data_out[in_dataset_idx]

        # TODO: Not yet able to run parameter sweeps on a method that also
        # produces mutliple output datasets
        if type(out_dataset) is list and current_param_sweep:
            err_str = (
                f"Parameter sweeps on methods with multiple output "
                f"datasets is not supported"
            )
            raise ValueError(err_str)

        # TODO: Not yet able to run a method that produces multiple output
        # datasets on input data that was the result of a paremeter sweep
        # earlier in the pipeline
        if (
            type(out_dataset) is list
            and type(dict_datasets_pipeline[in_dataset]) is list
        ):
            err_str = (
                f"Running a method that produces mutliple outputs on "
                f"the result of a parameter sweep is not supported"
            )
            raise ValueError(err_str)

        # Create a list to store the result of the different parameter values
        if current_param_sweep:
            out = []

        # Now, loop through all the arrays involved in the current method's
        # processing, taking into account several things:
        # - if reslicing needs to occur
        # - if global stats are needed by the method
        # - if extra parameters need to be added in order to handle the
        #   parameter sweep data (just `save_to_images()` falls into this
        #   category)
        for i, arr in enumerate(arrs):
            if method_name in SAVERS_NO_DATA_OUT_PARAM:
                # TODO: Assuming that the `save_to_images()` method is
                # being used when defining the `subfolder_name`
                # parameter value, as it is currently the only method in
                # `SAVERS_NO_DATA_OUT_PARAM`, not good...
                #
                # Define `subfolder_name` for `save_to_images()`
                subfolder_name = f"images_{i}"
                dict_params_method.update({"subfolder_name": subfolder_name})

            # Perform a reslice of the data if necessary
            if should_reslice:
                if reslice_info.reslice_dir is None:
                    resliced_data, _ = reslice(
                        arr,
                        current_slice_dim,
                        next_slice_dim,
                        comm,
                    )
                else:
                    resliced_data, _ = reslice_filebased(
                        arr,
                        current_slice_dim,
                        next_slice_dim,
                        comm,
                        reslice_info.reslice_dir,
                    )
                # Store the resliced input
                if type(dict_datasets_pipeline[in_dataset]) is list:
                    dict_datasets_pipeline[in_dataset][i] = resliced_data
                else:
                    dict_datasets_pipeline[in_dataset] = resliced_data
                arr = resliced_data

            dict_httomo_params["data"] = arr

            # Add global stats if necessary
            if req_glob_stats is True:
                stats = _fetch_glob_stats(arr, comm)
                glob_stats[in_dataset].append(stats)
                dict_params_method.update({"glob_stats": stats})

            # Run the method
            if method_name in SAVERS_NO_DATA_OUT_PARAM:
                _run_method_wrapper(
                    func_wrapper, method_name, dict_params_method, dict_httomo_params
                )
            else:
                if current_param_sweep:
                    for val in param_sweep_vals:
                        dict_params_method[param_sweep_name] = val
                        res = _run_method_wrapper(
                            func_wrapper,
                            method_name,
                            dict_params_method,
                            dict_httomo_params,
                        )
                        out.append(res)
                    dict_datasets_pipeline[out_dataset] = out
                else:
                    res = _run_method_wrapper(
                        func_wrapper,
                        method_name,
                        dict_params_method,
                        dict_httomo_params,
                    )
                    # Store the output(s) of the method in the appropriate
                    # dataset in the `dict_datasets_pipeline` dict
                    if type(res) in [list, tuple]:
                        # The method produced multiple outputs
                        for val, dataset in zip(res, out_dataset):
                            dict_datasets_pipeline[dataset] = val
                    else:
                        if type(dict_datasets_pipeline[out_dataset]) is list:
                            # Method has been run on an array that was part of a
                            # parameter sweep
                            dict_datasets_pipeline[out_dataset][i] = res
                        else:
                            dict_datasets_pipeline[out_dataset] = res

        if method_name in SAVERS_NO_DATA_OUT_PARAM:
            # Nothing more to do if the saver has a special kind of
            # output which handles saving the result
            return reslice_info, glob_stats

        # TODO: The dataset saving functionality only supports 3D data
        # currently, so check that the dimension of the data is 3 before
        # saving it
        is_3d = False
        # If `out_dataset` is a list, then this was a method which had a single
        # input and multiple outputs.
        #
        # TODO: For now, in this case, assume that none of the results need to
        # be saved, and instead will purely be used as inputs to other methods.
        if type(out_dataset) is list:
            # TODO: Not yet supporting parameter sweeps for methods that produce
            # multiple outputs, so can assume that if `out_datasets` is a list,
            # then no parameter sweep exists in the pipeline
            any_param_sweep = False
        elif type(dict_datasets_pipeline[out_dataset]) is list:
            # Either the method has had a parameter sweep, or been run on
            # parameter sweep input
            any_param_sweep = True
        else:
            # No parameter sweep is invovled, nor multiple output datasets, just
            # the simple case
            is_3d = len(dict_datasets_pipeline[out_dataset].shape) == 3
            any_param_sweep = False

        # Save the result if necessary
        if save_result and is_3d and not any_param_sweep:
            recon_algorithm = dict_params_method.pop("algorithm", None)
            if recon_algorithm is not None:
                slice_dim = 1
            else:
                slice_dim = _get_slicing_dim(current_func.pattern)

            intermediate_dataset(
                dict_datasets_pipeline[out_dataset],
                out_dir,
                comm,
                task_idx + 1,
                package_name,
                method_name,
                out_dataset,
                slice_dim,
                recon_algorithm=recon_algorithm,
            )
        elif save_result and any_param_sweep:
            # Save the result of each value in the parameter sweep as a
            # different dataset within the same hdf5 file, and also save the
            # middle slice of each parameter sweep result as a tiff file
            param_sweep_datasets = dict_datasets_pipeline[out_dataset]
            # For the output of a recon, fix the dimension that data is gathered
            # along to be the first one (ie, the vertical dim in volume space).
            # For all other types of methods, use the same dim associated to the
            # pattern for their input data.
            if RECON_MODULE_MATCH in module_path:
                slice_dim = 1
            else:
                slice_dim = _get_slicing_dim(current_func.pattern)
            data_shape = get_data_shape(param_sweep_datasets[0], slice_dim - 1)
            hdf5_file_name = f"{task_idx}-{package_name}-{method_name}-{out_dataset}.h5"
            # For each MPI process, send all the other processes the size of the
            # slice dimension of the parameter sweep arrays (note that the list
            # returned by `allgather()` is ordered by the rank of the process)
            all_proc_info = comm.allgather(param_sweep_datasets[i].shape[0])
            # Check if the current MPI process has the subset of data containing
            # the middle slice or not
            start_idx = 0
            for i in range(comm.rank):
                start_idx += all_proc_info[i]
            glob_mid_slice_idx = data_shape[slice_dim - 1] // 2
            has_mid_slice = (
                start_idx <= glob_mid_slice_idx
                and glob_mid_slice_idx < start_idx + param_sweep_datasets[0].shape[0]
            )

            # For the single MPI process that has access to the middle slices,
            # create an array to hold these middle slices
            if has_mid_slice:
                tiff_stack_shape = (
                    len(param_sweep_datasets),
                    data_shape[1],
                    data_shape[2],
                )
                middle_slices = np.empty(tiff_stack_shape)
                # Calculate the index relative to the subset of the data that
                # the MPI process has which corresponds to the middle slice of
                # the "global" data
                rel_mid_slice_idx = glob_mid_slice_idx - start_idx

            for i in range(len(param_sweep_datasets)):
                # Save hdf5 dataset
                dataset_name = f"/data/param_sweep_{i}"
                save_dataset(
                    out_dir,
                    hdf5_file_name,
                    param_sweep_datasets[i],
                    slice_dim=slice_dim,
                    chunks=(1, data_shape[1], data_shape[2]),
                    path=dataset_name,
                    comm=comm,
                )
                # Get the middle slice of the parameter-swept array
                if has_mid_slice:
                    if slice_dim == 1:
                        middle_slices[i] = param_sweep_datasets[i][
                            rel_mid_slice_idx, :, :
                        ]
                    elif slice_dim == 2:
                        middle_slices[i] = param_sweep_datasets[i][
                            :, rel_mid_slice_idx, :
                        ]
                    elif slice_dim == 3:
                        middle_slices[i] = param_sweep_datasets[i][
                            :, :, rel_mid_slice_idx
                        ]

            if has_mid_slice:
                # Save tiffs of the middle slices
                save_to_images(
                    middle_slices,
                    out_dir,
                    subfolder_name=f"middle_slices_{out_dataset}",
                )

    return reslice_info, glob_stats


def _run_loader(func_method: Callable, dict_params_method: Dict) -> LoaderData:
    """Run a loader function in the processing pipeline.

    Parameters
    ----------
    func_method : Callable
        The python function that performs the loading.
    dict_params_method : Dict
        A dict of parameters for the loader.

    Returns
    -------
    LoaderData
        loaded data that all loader functions return.
    """
    return func_method(**dict_params_method)


def _run_method_wrapper(
    func_wrapper: Callable,
    method_name: str,
    dict_params_method: Dict,
    dict_httomo_params: Dict,
) -> ndarray:
    """Run a wrapper method function (httomolib/tomopy) in the processing pipeline.

    Parameters
    ----------
    func_wrapper : Callable
        The wrapper function of the wrapper class that executes the method.
    method_name : str
        The name of the method to apply.
    dict_params_method : Dict
        A dict of parameters for the tomopy method.
    dict_httomo_params : Dict
        A dict of parameters related to httomo.

    Returns
    -------
    ndarray
        An array containing the result of the method function.
    """
    return func_wrapper(method_name, dict_params_method, **dict_httomo_params)


def _check_save_result(
    is_last: bool,
    module_path: str,
    save_all: bool,
    save_result_param: bool,
) -> bool:
    """Check if the result of the current method should be saved.

    Parameters
    ----------
    is_last : bool
        Whether the current task is last in the pipeline.
    module_path : str
        The path of the module that the method function comes from.
    save_all : bool
        Whether to save the result of all methods in the pipeline,
    save_result_param : bool
        The value of `save_result` for given method form the YAML (if not
        defined in the YAML, then this will have a value of `None`).

    Returns
    -------
    bool
        Whether or not to save the result of a method.

    """
    save_result = False

    # Default behaviour for saving datasets is to save the output of the last
    # task, and the output of reconstruction methods are always saved unless
    # specified otherwise.
    #
    # The default behaviour can be overridden in two ways:
    # 1. the flag `--save_all` which affects all tasks
    # 2. the method param `save_result` which affects individual tasks
    #
    # Here, enforce default behaviour.
    if is_last:
        save_result = True

    # Now, check if `--save_all` has been specified, as this can override
    # default behaviour
    if save_all:
        save_result = True

    # Now, check if it's a method from a reconstruction module
    if RECON_MODULE_MATCH in module_path:
        save_result = True

    # Finally, check if the `save_result` param was specified in the YAML
    # config, as it can override both default behaviour and the `--save_all`
    # flag
    if save_result_param is not None:
        save_result = save_result_param

    return save_result


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


def _set_method_data_param(
    func_method: Callable, dataset_name: str, datasets: Dict[str, ndarray]
) -> Dict[str, ndarray]:
    """Set a key in the param dict whose value is the array associated with the
    input dataset name given in the YAML config. The name of this key must be
    the same as the associated parameter name in the python function which
    performs the method, in order to pass the parameters via dict-unpacking.

    E.g. suppose a method function has parameters:

    def my_func(sino: ndarray, ...) -> ndarray:
        ...

    and in the user config YAML file, the dataset name to be passed into this
    method function is called `tomo`, defined by YAML like the following:
    ```
    my_func:
      data_in: tomo
      data_out: tomo_out
    ```

    This function `_set_method_data_param()` would return a dict that contains:
    - the key `sino` mapped to the value of the array associated to the `tomo`
      dataset

    Parameters
    ----------
    func_method : Callable
        The method function whose data parameters will be inspected.
    dataset_name : str
        The name of the input dataset name from the user config YAML.
    datasets : Dict[str, ndarray]
        A dict of all the available datasets in the current run.

    Returns
    -------
    Dict[str, ndarray]
        A dict containing the input data parameter key and value needed for the
        given method function.
    """
    sig_params = list(signature(func_method).parameters.keys())
    # For httomo functions, the input data paramater will always be the first
    # parameter
    data_param = sig_params[0]
    return {data_param: datasets[dataset_name]}


def _fetch_glob_stats(
    data: ndarray, comm: MPI.Comm
) -> Tuple[float, float, float, float]:
    """Fetch the min, max, mean, standard deviation of the given data.

    Parameters
    ----------
    data : ndarray
        The data to calculate statistics from.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[float, float, float, float]
        A tuple containing the stats values.
    """
    return min_max_mean_std(data, comm)


def _check_if_should_reslice(methods: List[MethodFunc]) -> List[MethodFunc]:
    """Determine if the input dataset for the method functions in the pipeline
    should be resliced. Builds the list of booleans.

    Parameters
    ----------
    methods : List[MethodFunc]
        List of the methods in the pipeline, associated with the patterns.

    Returns
    -------
    List[MethodFunc]
        Modified list of methods, with the ``reslice_ahead`` field set.
    """
    # ___________Rules for when and when-not to reslice the data___________
    # In order to reslice more accurately we need to know about all patterns in
    # the given pipeline.
    # The general rules are the following:
    # 1. Reslice ONLY if the pattern changes from "projection" to "sinogram" or the other way around
    # 2. With Pattern.all present one needs to check patterns on the edges of
    # the Pattern.all.
    # For instance consider the following example (method - pattern):
    #      1. Normalise - projection
    #      2. Dezinger - all
    #      3. Phase retrieval - projection
    #      4. Median - all
    #      5. Centering - sinogram
    # In this case you DON'T reclice between 2 and 3 as 1 and 3 are the same pattern.
    # You reclice between 4 and 5 as the pattern between 3 and 5 does change.
    total_number_of_methods = len(methods)
    ret_methods = [*methods]

    current_pattern = methods[0].pattern
    for x in range(total_number_of_methods):
        if (methods[x].pattern != current_pattern) and (
            methods[x].pattern != Pattern.all
        ):
            # skipping "all" pattern and look for different pattern from the
            # current pattern
            current_pattern = methods[x].pattern
            ret_methods[x] = dataclasses.replace(methods[x], reslice_ahead=True)

    return ret_methods


def _check_params_for_sweep(params: Dict) -> int:
    """Check the parameter dict of a method for the number of parameter sweeps
    that occur.
    """
    count = 0
    for k, v in params.items():
        if type(v) is tuple:
            count += 1
    return count


def _assign_pattern_to_method(method_func: MethodFunc) -> MethodFunc:
    """Fetch the pattern information from the methods database in
    `httomo/methods_database/packages` for the given method and associate that
    pattern with the function object.

    Parameters
    ----------
    method_func : MethodFunc
        The method function information whose pattern information will be fetched and populated.

    Returns
    -------
    MethodFunc
        The function information `pattern` attribute set, corresponding to the
        pattern that the method requires its input data to have.
    """
    pattern_str = get_method_info(
        method_func.module_name, method_func.method_function.__name__, "pattern"
    )
    if pattern_str == "projection":
        pattern = Pattern.projection
    elif pattern_str == "sinogram":
        pattern = Pattern.sinogram
    elif pattern_str == "all":
        pattern = Pattern.all
    else:
        err_str = (
            f"The pattern {pattern_str} that is listed for the method "
            f"{method_func.module_name} is invalid."
        )
        log_exception(err_str)
        raise ValueError(err_str)

    return dataclasses.replace(method_func, pattern=pattern)


def _determine_gpu_sections(method_funcs: List[MethodFunc]) -> List[PlatformSection]:
    ret: List[PlatformSection] = []
    current_gpu = method_funcs[0].gpu
    current_pattern = method_funcs[0].pattern
    methods: List[MethodFunc] = []
    for method in method_funcs:
        if (
            method.gpu == current_gpu
            and (
                method.pattern == current_pattern
                or method.pattern == Pattern.all
                or current_pattern == Pattern.all
            )
        ):
            methods.append(method)
            if current_pattern == Pattern.all and method.pattern != Pattern.all:
                current_pattern = method.pattern
        else:
            ret.append(
                PlatformSection(
                    gpu=current_gpu,
                    pattern=current_pattern,
                    max_slices=0,
                    methods=methods,
                )
            )
            methods = [method]
            current_pattern = method.pattern
            current_gpu = method.gpu

    ret.append(
        PlatformSection(
            gpu=current_gpu, pattern=current_pattern, max_slices=0, methods=methods
        )
    )

    return ret
