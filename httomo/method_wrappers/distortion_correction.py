from typing import Dict, Optional

from mpi4py.MPI import Comm

from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.preview import PreviewConfig
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
        return True

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        preview_config: PreviewConfig,
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
        self._update_params_from_preview(preview_config)

    def _update_params_from_preview(self, preview_config: PreviewConfig) -> None:
        """
        Extract information from preview config to define the parameter values required for
        distortion correction methods, and update `self._config_params`.
        """
        SHIFT_PARAM_NAME = "shift_xy"
        STEP_PARAM_NAME = "step_xy"
        shift_param_value = [
            preview_config.detector_x.start,
            preview_config.detector_y.start,
        ]
        step_param_value = [1, 1]
        self.append_config_params(
            {
                SHIFT_PARAM_NAME: shift_param_value,
                STEP_PARAM_NAME: step_param_value,
            }
        )
