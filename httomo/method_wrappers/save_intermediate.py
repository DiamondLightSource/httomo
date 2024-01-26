import os
import pathlib
from typing import Any, Dict, Optional
import weakref
from mpi4py.MPI import Comm
import httomo
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.dataset import DataSetBlock
from httomo.runner.loader import LoaderInterface
from httomo.runner.method_wrapper import MethodParameterDictType, MethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import xp

import h5py
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
                 out_dir: os.PathLike = httomo.globals.run_out_dir,
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
        
        self._file = h5py.File(f"{out_dir}/{filename}.h5", "w", driver="mpio", comm=comm)
        # make sure file gets closed properly
        weakref.finalize(self, self._file.close)
        
    def _transform_params(self, dict_params: MethodParameterDictType) -> MethodParameterDictType:
        dict_params = super()._transform_params(dict_params).copy()
        dict_params["detector_x"] = self._loader.detector_x
        dict_params["detector_y"] = self._loader.detector_y
        dict_params["path"] = "/data"
        dict_params["file"] = self._file
        return dict_params
    
    def _process_return_type(self, ret: Any, input_dataset: DataSetBlock) -> DataSetBlock:
        if input_dataset.is_last_in_chunk:
            self._file.close()
        return input_dataset

        