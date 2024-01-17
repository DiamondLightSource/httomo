import httomo.globals
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.dataset import DataSetBlock
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import _get_slicing_dim, xp


from mpi4py.MPI import Comm


import os
from typing import Dict, Optional


class ImagesWrapper(GenericMethodWrapper):
    """Wraps image writer methods, which accept numpy (CPU) arrays as input,
    but don't actually modify the dataset. They write the information to files"""

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return module_path.endswith(".images")

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        save_result: Optional[bool] = None,
        output_mapping: Dict[str, str] = {},
        out_dir: Optional[os.PathLike] = None,
        **kwargs,
    ):
        super().__init__(
            method_repository,
            module_path,
            method_name,
            comm,
            save_result,
            output_mapping,
            **kwargs,
        )
        self["out_dir"] = out_dir if out_dir is not None else httomo.globals.run_out_dir
        if "comm_rank" in self.parameters:
            raise ValueError(
                "save_to_images with the comm_rank parameter is broken. " +
                "Please upgrade to the latest version, taking an offset parameter"
            )

    # Images execute is leaving original data on the device where it is,
    # but gives the method a CPU copy of the data.
    def execute(
        self,
        dataset: DataSetBlock,
    ) -> DataSetBlock:
        config_params = self._config_params
        if "offset" in self.parameters:
            config_params = {
                **self._config_params,
                "offset": dataset.global_index[_get_slicing_dim(self.pattern) - 1],
            }
            
        args = self._build_kwargs(self._transform_params(config_params), dataset)
        if dataset.is_gpu:
            # give method a CPU copy of the data
            args[self.parameters[0]] = xp.asnumpy(dataset.data)

        self.method(**args)

        return dataset