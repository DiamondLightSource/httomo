import os
import pathlib
from typing import Any, Dict, Optional, Union
import weakref
from mpi4py.MPI import Comm
import httomo
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.dataset import DataSetBlock
from httomo.runner.loader import LoaderInterface
from httomo.runner.method_wrapper import GpuTimeInfo, MethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import catchtime, xp

import h5py
import zarr
import numpy as np


class SaveIntermediateFilesWrapper(GenericMethodWrapper):
    
    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return method_name == "save_intermediate_data"
    
    def __init__(self, 
                 method_repository: MethodRepository, 
                 module_path: str, 
                 method_name: str, 
                 comm: Comm, 
                 save_result: Optional[bool] = None,
                 output_mapping: Dict[str, str] = {}, 
                 out_dir: Optional[os.PathLike] = None,
                 prev_method: Optional[MethodWrapper] = None,
                 loader: Optional[LoaderInterface] = None,
                 **kwargs):
        super().__init__(method_repository, module_path, method_name, comm, save_result, output_mapping, **kwargs)
        assert loader is not None
        self._loader = loader
        assert prev_method is not None

        filename = f"{prev_method.task_id}-{prev_method.package_name}-{prev_method.method_name}"
        if prev_method.recon_algorithm is not None:
            filename += f"-{prev_method.recon_algorithm}"
        
        if out_dir is None:
            out_dir = httomo.globals.run_out_dir
        assert out_dir is not None

        self._file: Union[h5py.File, zarr.Group]
        if httomo.globals.INTERMEDIATE_FORMAT == "hdf5":
            self._file = h5py.File(
                f"{out_dir}/{filename}.h5", "w", driver="mpio", comm=comm
            )
            # make sure hdf5 file gets closed properly
            weakref.finalize(self, self._file.close)
        else:
            synchronizer = zarr.ProcessSynchronizer(
                pathlib.Path(out_dir, f"{filename}.sync")
            )
            self._file = zarr.open_group(
                store=f"{out_dir}/{filename}.zarr",
                mode="a",
                synchronizer=synchronizer,
            )

    def execute(self, block: DataSetBlock) -> DataSetBlock:
        # we overwrite the whole execute method here, as we do not need any of the helper
        # methods from the Generic Wrapper
        # What we know:
        # - we do not transfer the dataset as a whole to CPU - only the data and angles locally
        # (and never back)
        # - the user does not insert this method - it's automatic - so no config params are
        # relevant
        # - we return just the input as it is
        self._gpu_time_info = GpuTimeInfo()
        with catchtime() as t:
            data = block.data if block.is_cpu else xp.asnumpy(block.data)
        self._gpu_time_info.device2host += t.elapsed

        self._method(
            data,
            global_shape=block.global_shape,
            global_index=block.global_index,
            slicing_dim=block.slicing_dim,
            file=self._file,
            path="data",
            detector_x=self._loader.detector_x,
            detector_y=self._loader.detector_y,
            angles=block.angles,
        )
        
        if block.is_last_in_chunk and isinstance(self._file, h5py.File):
            self._file.close()

        return block
