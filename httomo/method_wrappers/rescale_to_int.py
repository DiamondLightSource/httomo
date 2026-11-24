from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.method_wrapper import MethodParameterDictType
from httomo.runner.methods_repository_interface import MethodRepository

from mpi4py.MPI import Comm
from typing import Dict, Optional


class RescaleToIntWrapper(GenericMethodWrapper):
    """
    Wraps the rescale_to_int method, to accept hardcoded values given in the pipeline file.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return method_name == "rescale_to_int"

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        save_result: Optional[bool] = None,
        output_mapping: Dict[str, str] = {},
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

        assert method_name == "rescale_to_int", "Only rescale_to_int is supported"

    def _transform_params(
        self, dict_params: MethodParameterDictType
    ) -> MethodParameterDictType:
        if (glob_stats := dict_params.get("glob_stats")) is not None and isinstance(
            glob_stats, list
        ):
            dict_params["glob_stats"] = tuple(glob_stats)

        return dict_params
