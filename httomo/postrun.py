"""
Module containing postrun functionality for the methods in the HTTomo pipeline.
This includes saving the results - hdf5 datasets, tiff images, etc.
"""
from typing import Any, Dict, Optional

import numpy as np
from httomolib.misc.images import save_to_images
from mpi4py import MPI

import httomo.globals
from httomo.common import MethodFunc, RunMethodInfo
from httomo.data.hdf._utils.chunk import get_data_shape, save_dataset
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo.utils import _get_slicing_dim

comm = MPI.COMM_WORLD


def postrun_method(
    run_method_info: RunMethodInfo,
    out_dataset: Any,
    dict_datasets_pipeline: Dict[str, Optional[np.ndarray]],
    current_func: MethodFunc,
    i: int,
):
    is_3d = False
    # If `out_dataset` is a list, then this was a method which had a single
    # input and multiple outputs.
    #
    # TODO: For now, in this case, assume that none of the results need to
    # be saved, and instead will purely be used as inputs to other methods.
    if isinstance(out_dataset, list):
        # TODO: Not yet supporting parameter sweeps for methods that produce
        # multiple outputs, so can assume that if `out_datasets` is a list,
        # then no parameter sweep exists in the pipeline
        any_param_sweep = False
    elif isinstance(dict_datasets_pipeline[out_dataset], list):
        # Either the method has had a parameter sweep, or been run on
        # parameter sweep input
        any_param_sweep = True
    else:
        # No parameter sweep is invovled, nor multiple output datasets, just
        # the simple case
        is_3d = len(dict_datasets_pipeline[out_dataset].shape) == 3
        any_param_sweep = False

    # Save the result if necessary
    if run_method_info.save_result and is_3d and not any_param_sweep:
        recon_center = run_method_info.dict_params_method.pop("center", None)
        recon_algorithm = None
        if recon_center is not None:
            slice_dim = 1
            recon_algorithm = run_method_info.dict_params_method.pop(
                "algorithm", None
            )  # covers tomopy case
        else:
            slice_dim = _get_slicing_dim(current_func.pattern)

        intermediate_dataset(
            dict_datasets_pipeline[out_dataset],
            httomo.globals.run_out_dir,
            comm,
            run_method_info.task_idx + 1,
            run_method_info.package_name,
            run_method_info.method_name,
            out_dataset,
            slice_dim,
            recon_algorithm=recon_algorithm,
        )
    elif run_method_info.save_result and any_param_sweep:
        # Save the result of each value in the parameter sweep as a
        # different dataset within the same hdf5 file, and also save the
        # middle slice of each parameter sweep result as a tiff file
        param_sweep_datasets = dict_datasets_pipeline[out_dataset]
        # For the output of a recon, fix the dimension that data is gathered
        # along to be the first one (ie, the vertical dim in volume space).
        # For all other types of methods, use the same dim associated to the
        # pattern for their input data.
        if "recon.algorithm" in current_func.module_name:
            slice_dim = 1
        else:
            slice_dim = _get_slicing_dim(current_func.pattern)
        data_shape = get_data_shape(param_sweep_datasets[0], slice_dim - 1)
        hdf5_file_name = f"{run_method_info.task_idx}-{run_method_info.package_name}-{run_method_info.method_name}-{out_dataset}.h5"
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
                httomo.globals.run_out_dir,
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
                    middle_slices[i] = param_sweep_datasets[i][rel_mid_slice_idx, :, :]
                elif slice_dim == 2:
                    middle_slices[i] = param_sweep_datasets[i][:, rel_mid_slice_idx, :]
                elif slice_dim == 3:
                    middle_slices[i] = param_sweep_datasets[i][:, :, rel_mid_slice_idx]

        if has_mid_slice:
            # Save tiffs of the middle slices
            save_to_images(
                middle_slices,
                httomo.globals.run_out_dir,
                subfolder_name=f"middle_slices_{out_dataset}",
            )
