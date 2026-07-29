from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.block_interfaces import T


class Sino360to180Wrapper(GenericMethodWrapper):
    """
    Wrapper to perform extended FoV (360degrees) data conversion to a standard 180 degrees data.
    The wrapper is responsible for changing the angles after the data has changed.
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return "sino_360_to_180" in method_name

    def _postprocess_data(self, block: T) -> T:
        # for 360 degrees data the angular dimension is truncated so the angles should be changed in a similar fashion.
        block.angles_radians = block.angles_radians[0 : block.data.shape[0]]
        return block
