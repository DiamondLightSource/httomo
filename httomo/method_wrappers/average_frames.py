from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.block_interfaces import T
import numpy as np


class AverageFramesWrapper(GenericMethodWrapper):
    """
    Wrapper for frames/projection averaging.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return "average_projection_frames" in method_name

    def _preprocess_data(self, block: T) -> T:
        # when the angular preview is getting changed by averaging the angles should be changed accordingly
        config_params = self._config_params
        k = config_params["projection_averaging_factor"]

        n_proj = block.data.shape[0]  # original data angular size
        n_full = n_proj // k
        remainder = n_proj % k

        n_out = n_full + (remainder > 0)

        averaged_angles = np.empty(n_out, dtype=block.angles.dtype)

        if n_full:
            averaged_angles[:n_full] = (
                block.angles_radians[: n_full * k]
                .reshape(n_full, k)
                .mean(axis=1)
            )

        if remainder:
            averaged_angles[-1] = block.angles_radians[n_full * k :].mean()

        block.angles_radians = averaged_angles
        return block
