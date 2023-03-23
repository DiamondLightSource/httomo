import multiprocessing
from datetime import datetime
from os import mkdir
from pathlib import Path
import time
from typing import List, Dict, Optional, Tuple, Union
from collections.abc import Callable
from inspect import signature
from importlib import import_module

from numpy import ndarray
from mpi4py import MPI

from httomo.utils import print_once, Pattern, _get_slicing_dim, Colour
from httomo.yaml_utils import open_yaml_config
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo.data.hdf._utils.reslice import reslice, reslice_filebased
from httomo._stats.globals import min_max_mean_std
from httomo.methods_database.query import get_method_info

from httomo.wrappers_class import TomoPyWrapper
from httomo.wrappers_class import HttomolibWrapper


def run_tasks(
    in_file: Path,
    yaml_config: Path,
    out_dir: Path,
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
    out_dir : Path
        The directory to write data to.
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
    run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )
    if comm.rank == 0:
        mkdir(run_out_dir)
    if comm.size == 1:
        ncore = (
            multiprocessing.cpu_count()
        )  # use all available CPU cores if not an MPI run

    # TODO: Define a list of savers which have no output dataset and so need to
    # be treated differently to other methods. Probably should be handled in a
    # more robust way than this workaround.
    SAVERS_NO_DATA_OUT_PARAM = ["save_to_images"]

    # Define dict to store arrays that are specified as datasets in the user
    # config YAML
    datasets = _initialise_datasets(yaml_config, SAVERS_NO_DATA_OUT_PARAM)

    # Define list to store dataset stats for each task in the user config YAML
    glob_stats = _initialise_stats(yaml_config)

    # Get a list of the python functions associated to the methods defined in
    # user config YAML
    method_funcs = _get_method_funcs(yaml_config, comm)

    # Define dict of params that are needed by loader functions
    loader_extra_params = {
        "in_file": in_file,
        "dimension": dimension,
        "pad": pad,
        "comm": comm,
    }

    # Hardcoded string that is used to check if a method is in a reconstruction
    # module or not
    RECON_MODULE_MATCH = "recon.algorithm"
    # A counter to track how many reslices occur in the processing pipeline
    reslice_counter = 0
    reslice_warn_str = (
        f"WARNING: Reslicing is performed more than once in "
        f"this pipeline, is there a need for this?"
    )
    has_reslice_warn_printed = False

    # Associate patterns to method function objects
    for i, (module_path, func, func_runner, params, is_loader) in enumerate(
        method_funcs
    ):
        func = _assign_pattern_to_method(func, module_path, params["method_name"])
        method_funcs[i] = (module_path, func, func_runner, params, is_loader)

    # get a list with booleans to identify when reslicing needed (True) or not
    # (False).
    patterns = [f.pattern for (_, f, _, _, _) in method_funcs]
    reslice_bool_list = _check_if_should_reslice(patterns)

    # start MPI timer for rank 0
    if comm.rank == 0:
        start_time = MPI.Wtime()

    # Run the methods
    for idx, (module_path, func, func_runner, params, is_loader) in enumerate(
        method_funcs
    ):
        package = module_path.split(".")[0]
        method_name = params.pop("method_name")
        task_no_str = f"Running task {idx+1}"
        task_end_str = task_no_str.replace("Running", "Finished")
        pattern_str = f"(pattern={func.pattern.name})"
        print_once(
            f"{task_no_str} {pattern_str}: {method_name}...",
            comm,
            colour=Colour.LIGHT_BLUE,
        )
        start = time.perf_counter_ns()
        if is_loader:
            params.update(loader_extra_params)

            # Check if a value for the `preview` parameter of the loader has
            # been provided
            if "preview" not in params.keys():
                params["preview"] = [None]

            (
                data,
                flats,
                darks,
                angles,
                angles_total,
                detector_y,
                detector_x,
            ) = _run_loader(func, params)

            # Update `datasets` dict with the data that has been loaded by the
            # loader
            datasets[params["name"]] = data
            datasets["flats"] = flats
            datasets["darks"] = darks

            # Define all params relevant to httomo that a wrapper function might
            # need
            possible_extra_params = [
                (["darks"], darks),
                (["flats"], flats),
                (["angles", "angles_radians"], angles),
                (["comm"], comm),
                (["out_dir"], run_out_dir),
            ]
        else:
            save_result = False

            # Default behaviour for saving datasets is to save the output of the
            # last task, and the output of reconstruction methods are always
            # saved unless specified otherwise.
            #
            # The default behaviour can be overridden in two ways:
            # 1. the flag `--save_all` which affects all tasks
            # 2. the method param `save_result` which affects individual tasks
            #
            # Here, enforce default behaviour.
            if idx == len(method_funcs) - 1:
                save_result = True

            # Now, check if `--save_all` has been specified, as this can
            # override default behaviour
            if save_all:
                save_result = True

            # Now, check if it's a method from a reconstruction module
            if RECON_MODULE_MATCH in module_path:
                save_result = True

            # Finally, check if `save_result` param has been specified in the
            # YAML config, as it can override both default behaviour and the
            # `--save_all` flag
            if "save_result" in params.keys():
                save_result = params.pop("save_result")

            # Check if the input dataset should be resliced before the task runs
            should_reslice = reslice_bool_list[idx]
            if should_reslice:
                reslice_counter += 1
                current_slice_dim = _get_slicing_dim(method_funcs[idx - 1][1].pattern)
                next_slice_dim = _get_slicing_dim(func.pattern)

            # the GPU wrapper should be aware if the reslice is needed to convert the result to numpy
            reslice_ahead = (
                reslice_bool_list[idx + 1]
                if idx < len(reslice_bool_list) - 1
                else "False"
            )
            if idx == 1:
                possible_extra_params.append((["reslice_ahead"], reslice_ahead))
            else:
                possible_extra_params[-1] = (["reslice_ahead"], reslice_ahead)

            if reslice_counter > 1 and not has_reslice_warn_printed:
                print_once(reslice_warn_str, comm=comm, colour=Colour.RED)
                has_reslice_warn_printed = True

            # Check for any extra params unrelated to wrapped packages but related to
            # httomo that should be added in
            httomo_params = _check_signature_for_httomo_params(
                func_runner, method_name, possible_extra_params
            )

            # Get the information describing if the method is being run only
            # once, or multiple times with different input datasets
            #
            # Make the input and output datasets always be lists just to be
            # generic, and further down loop through all the datasets that the
            # method should be applied to
            if "data_in" in params.keys() and "data_out" in params.keys():
                data_in = [params.pop("data_in")]
                data_out = [params.pop("data_out")]
            elif "data_in_multi" in params.keys() and "data_out_multi" in params.keys():
                data_in = params.pop("data_in_multi")
                data_out = params.pop("data_out_multi")
            else:
                # TODO: This error reporting is possibly better handled by
                # schema validation of the user config YAML
                if method_name not in SAVERS_NO_DATA_OUT_PARAM:
                    err_str = "Invalid in/out dataset parameters"
                    raise ValueError(err_str)
                else:
                    data_in = [params.pop("data_in")]
                    data_out = [None]

            # Check if the method function's params require any datasets stored
            # in the `datasets` dict
            dataset_params = _check_method_params_for_datasets(params, datasets)
            # Update the relevant parameter values according to the required
            # datasets
            params.update(dataset_params)

            # check if the module needs the ncore parameter and add it
            if "ncore" in signature(func).parameters:
                params.update({"ncore": ncore})
            # check if the module needs the gpu_id parameter and flag it to add in the wrapper
            gpu_id_par = "gpu_id" in signature(func).parameters
            if gpu_id_par:
                params.update({"gpu_id": gpu_id_par})

            # Check if method type signature requires global statistics
            req_glob_stats = "glob_stats" in signature(func_runner).parameters

            for i, in_dataset in enumerate(data_in):
                if save_result:
                    out_dir = run_out_dir
                else:
                    out_dir = None

                if should_reslice:
                    if reslice_dir is None:
                        resliced_data, _ = reslice(
                            datasets[in_dataset],
                            current_slice_dim,
                            next_slice_dim,
                            comm,
                        )
                    else:
                        resliced_data, _ = reslice_filebased(
                            datasets[in_dataset],
                            current_slice_dim,
                            next_slice_dim,
                            comm,
                            reslice_dir,
                        )
                    datasets[in_dataset] = resliced_data

                if req_glob_stats is True:
                    stats = _fetch_glob_stats(datasets[in_dataset], comm)
                    glob_stats[idx][in_dataset] = stats
                    params.update({"glob_stats": stats})

                _run_method(
                    func_runner,
                    idx + 1,
                    package,
                    method_name,
                    in_dataset,
                    data_out[i],
                    datasets,
                    params,
                    httomo_params,
                    SAVERS_NO_DATA_OUT_PARAM,
                    comm,
                    out_dir=out_dir,
                )
        stop = time.perf_counter_ns()
        print_once(
            f"{task_end_str} {pattern_str}: {method_name} ({package}): Took {float(stop-start)*1e-6:.2f}ms",
            comm,
        )

        stop = time.perf_counter_ns()
        output_str_list = [
            f"{task_end_str} {pattern_str}: {method_name} (",
            package,
            f") Took {float(stop-start)*1e-6:.2f}ms",
        ]
        output_colour_list = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
        print_once(output_str_list, comm=comm, colour=output_colour_list)

    # Print the number of reslice operations peformed in the pipeline
    reslice_summary_str = f"Total number of reslices: {reslice_counter}"
    reslice_summary_colour = Colour.BLUE if reslice_counter <= 1 else Colour.RED
    print_once(reslice_summary_str, comm=comm, colour=reslice_summary_colour)

    elapsed_time = 0
    if comm.rank == 0:
        elapsed_time = MPI.Wtime() - start_time
        end_str = f"\n\n~~~ Pipeline finished ~~~\nTook {elapsed_time} sec to run!"
        print_once(end_str, comm=comm, colour=Colour.BVIOLET)


def _initialise_datasets(
    yaml_config: Path, savers_no_data_out_param: List[str]
) -> Dict[str, None]:
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
    datasets = {}
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
def _initialise_stats(yaml_config: Path) -> List[Dict]:
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
    stats = []
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
        method_stats = {}

        # Check if there are multiple input datasets to account for
        if type(method_conf[dataset_param]) is list:
            for dataset_name in method_conf[dataset_param]:
                method_stats[dataset_name] = None
        else:
            method_stats[method_conf[dataset_param]] = None

        stats.append(method_stats)

    return stats


def _get_method_funcs(
    yaml_config: Path, comm: MPI.Comm
) -> List[Tuple[str, Callable, Dict, bool]]:
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
    List[Tuple[Callable, Dict, bool]]
        A list, each element being a tuple containing four elements:
        - a package name
        - a method function
        - a dict of parameters for the method function
        - a boolean describing if it is a loader function or not
    """
    method_funcs = []
    yaml_conf = open_yaml_config(yaml_config)

    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        split_module_name = module_name.split(".")
        method_name, method_conf = module_conf.popitem()
        method_conf["method_name"] = method_name

        if split_module_name[0] == "httomo":
            # deal with httomo loaders
            if "loaders" in module_name:
                is_loader = True
            else:
                is_loader = False
            module = import_module(module_name)
            method_func = getattr(module, method_name)
            method_funcs.append(
                (module_name, method_func, None, method_conf, is_loader)
            )
        elif (split_module_name[0] == "tomopy") or (
            split_module_name[0] == "httomolib"
        ):
            if split_module_name[0] == "tomopy":
                # initialise the TomoPy wrapper class
                wrapper_init_module = TomoPyWrapper(
                    split_module_name[1], split_module_name[2], method_name, comm
                )
            if split_module_name[0] == "httomolib":
                # initialise the httomolib wrapper class
                wrapper_init_module = HttomolibWrapper(
                    split_module_name[1], split_module_name[2], method_name, comm
                )
            wrapper_func = getattr(wrapper_init_module.module, method_name)
            wrapper_method = wrapper_init_module.wrapper_method
            method_funcs.append(
                (module_name, wrapper_func, wrapper_method, method_conf, False)
            )
        else:
            err_str = (
                f"An unknown module name was encountered: " f"{split_module_name[0]}"
            )
            raise ValueError(err_str)

    return method_funcs


def _run_method(
    func_runner: Callable,
    task_no: int,
    package_name: str,
    method_name: str,
    in_dataset: str,
    out_dataset: Union[str, List[str]],
    datasets: Dict[str, ndarray],
    method_params: Dict,
    httomo_params: Dict,
    savers_no_data_out_param: List[str],
    comm: MPI.Comm,
    out_dir: str = None,
) -> ndarray:
    """Run a method function in the processing pipeline.

    Parameters
    ----------
    func_runner : Callable
        The python function that performs the method.
    task_no : int
        The number of the given task, starting at index 1.
    package_name : str
        The package that the method function `func` comes from.
    method_name : str
        The name of the method to apply.
    in_dataset : str
        The name of the input dataset.
    out_dataset : Union[str, List[str]]
        The name(s) of the output dataset(s).
    datasets : Dict[str, ndarray]
        A dict containing all available datasets in the given pipeline.
    method_params : Dict
        A dict of parameters for the method.
    httomo_params : Dict, optional
        A dict of parameters related to HTTomo.
    savers_no_data_out_param : List[str]
        A list of savers which have neither `data_out` nor `data_out_multi` as
        their output.
    comm : MPI.Comm
        MPI communicator object.
    out_dir : str, optional
        If the result should be saved in an intermediate file, the directory to
        save it should be provided.

    Returns
    -------
    ndarray
        An array containing the result of the method function.
    """
    # Add the appropriate dataset to the method function's dict of
    # parameters based on the parameter name for the method's python
    # function
    if package_name in ["httomolib", "tomopy"]:
        httomo_params["data"] = datasets[in_dataset]

    if method_name in savers_no_data_out_param:
        _run_method_wrapper(func_runner, method_name, method_params, httomo_params)
        # Nothing more to do with output data if the saver has a special
        # kind of output
        return
    else:
        res = _run_method_wrapper(
            func_runner, method_name, method_params, httomo_params
        )

    # Store the output(s) of the method in the appropriate dataset in the
    # `datasets` dict
    if type(res) in [list, tuple]:
        for val, dataset in zip(res, out_dataset):
            datasets[dataset] = val
    else:
        datasets[out_dataset] = res

    # TODO: The dataset saving functionality only supports 3D data
    # currently, so check that the dimension of the data is 3 before
    # saving it
    is_3d = False
    # If `out_dataset` is a list, then this was a method which had a single
    # input and multiple outputs.
    #
    # TODO: For now, in this case, assume that none of the results need to be
    # saved, and instead will purely be used as inputs to other methods.
    if not isinstance(out_dataset, list):
        is_3d = len(datasets[out_dataset].shape) == 3
    # Save the result if necessary
    if out_dir is not None and is_3d:
        # Check the slice dim for the method, so then the data from the
        # different MPI processes can be gathered along the correct axis when
        # saving to an intermediate file
        recon_algorithm = method_params.pop("algorithm", None)
        if recon_algorithm is not None:
            slice_dim = 1
        else:
            slice_dim = _get_slicing_dim(func.pattern)

        intermediate_dataset(
            datasets[out_dataset],
            out_dir,
            comm,
            task_no,
            package_name,
            method_name,
            out_dataset,
            slice_dim,
            recon_algorithm=recon_algorithm,
        )


def _run_loader(
    func: Callable, params: Dict
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int, int, int]:
    """Run a loader function in the processing pipeline.

    Parameters
    ----------
    func : Callable
        The python function that performs the loading.
    params : Dict
        A dict of parameters for the loader.

    Returns
    -------
    Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int, int, int]
        A tuple of 8 values that all loader functions return.
    """
    return func(**params)


def _run_method_wrapper(
    func_runner: Callable, method_name: str, method_params: Dict, httomo_params: Dict
) -> ndarray:
    """Run a wrapper method function (httomolib/tomopy) in the processing pipeline.

    Parameters
    ----------
    func_runner : Callable
        The python function that performs the method.
    method_name : str
        The name of the method to apply.
    method_params : Dict
        A dict of parameters for the tomopy method.
    httomo_params : Dict
        A dict of parameters related to httomo.

    Returns
    -------
    ndarray
        An array containing the result of the method function.
    """
    return func_runner(method_name, method_params, **httomo_params)


def _check_signature_for_httomo_params(
    func_runner: Callable, method_name: str, params: List[Tuple[List[str], object]]
) -> Dict:
    """Check if the given method requires any parameters related to HTTomo.

    Parameters
    ----------
    func_runner : Callable
        Function whose type signature is to be inspected
    method_name : str
        The name of the method to apply.

    params : List[Tuple[List[str], object]]
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
    sig_params = signature(func_runner).parameters
    for names, val in params:
        for name in names:
            if name in sig_params:
                extra_params[name] = val
    return extra_params


def _check_method_params_for_datasets(
    params: Dict, datasets: Dict[str, ndarray]
) -> Dict:
    """Check a given method function's parameter values to see if any of them
    are a dataset.

    Parameters
    ----------
    params : Dict
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
    for name, val in params.items():
        # If the value of this parameter is a dataset, then replace the
        # parameter value with the dataset value
        if val in datasets:
            dataset_params[name] = datasets[val]
    return dataset_params


def _set_method_data_param(
    func: Callable, dataset_name: str, datasets: Dict[str, ndarray]
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
    func : Callable
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
    sig_params = list(signature(func).parameters.keys())
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


def _check_if_should_reslice(patterns: List[Pattern]) -> List[bool]:
    """Determine if the input dataset for the method functions in the pipeline
    should be resliced. Builds the list of booleans.

    Parameters
    ----------
    patterns : List[Pattern]
        List of the patterns associated with the python functions needed for the
        run.

    Returns
    -------
    List[bool]
        List with booleans which methods need reslicing (True) or not (False).
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
    total_number_of_methods = len(patterns)
    reslice_bool_list = [False] * total_number_of_methods

    current_pattern = patterns[0]
    for x in range(total_number_of_methods):
        if (patterns[x] != current_pattern) and (patterns[x] != Pattern.all):
            # skipping "all" pattern and look for different pattern from the
            # current pattern
            current_pattern = patterns[x]
            reslice_bool_list[x] = True
    return reslice_bool_list


def _assign_pattern_to_method(
    func: Callable, module_path: str, method_name: str
) -> Callable:
    """Fetch the pattern information from the methods database in
    `httomo/methods_database/packages` for the given method and associate that
    pattern with the function object.

    Parameters
    ----------
    func : Callable
        The method function whose pattern information will be fetched.

    module_path : str
        The module path to the method function.

    method_name : str
        The name of the method function.

    Returns
    -------
    Callable
        The function object with a `.pattern` attribute it corresponding to the
        pattern that the method requires its input data to have.
    """
    pattern_str = get_method_info(module_path, method_name, "pattern")
    if pattern_str == "projection":
        pattern = Pattern.projection
    elif pattern_str == "sinogram":
        pattern = Pattern.sinogram
    elif pattern_str == "all":
        pattern = Pattern.all
    else:
        err_str = (
            f"The pattern {pattern_str} that is listed for the method "
            f"{module_path} is invalid."
        )
        raise ValueError(err_str)

    func.pattern = pattern
    return func
