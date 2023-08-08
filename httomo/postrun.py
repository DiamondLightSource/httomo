"""
Module containing postrun functionality for the methods in the HTTomo pipeline.
This includes saving the results - hdf5 datasets, tiff images, etc.
"""
from typing import Any, Dict, Optional

import numpy as np
from httomolib.misc.images import save_to_images
from mpi4py import MPI

import httomo.globals
from httomo.common import MethodFunc, RunMethodInfo, PlatformSection
from httomo.data.hdf._utils.chunk import get_data_shape, save_dataset
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo.utils import _get_slicing_dim

comm = MPI.COMM_WORLD


def postrun_method(
    run_method_info: RunMethodInfo,
    dict_datasets_pipeline: Dict[str, Optional[np.ndarray]],
    section: PlatformSection,
):
    if run_method_info.method_name != "save_to_images" and 'center' not in run_method_info.method_name:
    
        is_3d = len(dict_datasets_pipeline[run_method_info.data_out].shape) == 3

        # Save the result into intermediate file if necessary
        if run_method_info.save_result and is_3d:
            recon_center = run_method_info.dict_params_method.pop("center", None)
            recon_algorithm = None
            if recon_center is not None:
                # get the name of the reconstruction algorithm
                recon_algorithm = run_method_info.dict_params_method.pop(
                    "algorithm", None
                )

            slice_dim = _get_slicing_dim(section.pattern)

            intermediate_dataset(
                dict_datasets_pipeline[run_method_info.data_out],
                httomo.globals.run_out_dir,
                comm,
                run_method_info.task_idx_global,
                run_method_info.package_name,
                run_method_info.method_name,
                run_method_info.data_out,
                slice_dim,
                recon_algorithm=recon_algorithm,
            )
