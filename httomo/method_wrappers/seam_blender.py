from typing import Dict, Optional

from mpi4py.MPI import Comm

from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.preview import PreviewConfig
from httomo.runner.methods_repository_interface import MethodRepository


class SeamBlenderWrapper(GenericMethodWrapper):
    """
    Wrapper for seam blender.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return "seam_blend" in method_name

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
        Extract information from preview config to modify seam index parameter required for `seam_blend_stitched_data` method.
        """
        SHIFT_PARAM_NAME = "shift_seam_index"
        det_x_start = preview_config.detector_x.start

        self.append_config_params(
            {
                SHIFT_PARAM_NAME: det_x_start,
            }
        )
