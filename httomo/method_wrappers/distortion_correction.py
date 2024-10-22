from typing import Dict, Optional

from mpi4py.MPI import Comm

from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository


class DistortionCorrectionWrapper(GenericMethodWrapper):
    """
    Wrapper for distortion correction methods.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return "distortion_correction" in method_name

    @classmethod
    def requires_preview(cls) -> bool:
        """
        Whether the wrapper class needs the preview information from the loader to execute the
        methods it wraps or not.
        """
        return True

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
