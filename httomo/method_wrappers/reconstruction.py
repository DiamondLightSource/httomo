from httomo.block_interfaces import T, Block
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.method_wrapper import MethodParameterDictType


from typing import Any, Dict, Optional

from httomo_backends.methods_database.query import Pattern


class ReconstructionWrapper(GenericMethodWrapper):
    """Wraps reconstruction functions, limiting the length of the angles array
    before calling the method."""

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return module_path.endswith(".algorithm")

    def _preprocess_data(self, block: T) -> T:
        # this is essential for the angles cutting below to be valid
        assert (
            self.pattern == Pattern.sinogram
        ), "reconstruction methods must be sinogram"

        # for 360 degrees data the angular dimension will be truncated while angles are not.
        # Truncating angles if the angular dimension has got a different size
        datashape0 = block.data.shape[0]
        if datashape0 != len(block.angles_radians):
            block.angles_radians = block.angles_radians[0:datashape0]
        self._input_shape = block.data.shape
        return super()._preprocess_data(block)

    def _build_kwargs(
        self,
        dict_params: MethodParameterDictType,
        dataset: Optional[Block] = None,
    ) -> Dict[str, Any]:
        assert dataset is not None, "Reconstruction wrappers require a dataset"
        # for recon methods, we assume that the second parameter is the angles in all cases
        assert (
            len(self.parameters) >= 2
        ), "recon methods always take data + angles as the first 2 parameters"
        updated_params = {**dict_params, self.parameters[1]: dataset.angles_radians}
        return super()._build_kwargs(updated_params, dataset)

    @property
    def recon_algorithm(self) -> Optional[str]:
        assert "center" in self.parameters, (
            "All recon methods should have a 'center' parameter, but it doesn't seem"
            + f" to be the case for {self.module_path}.{self.method_name}"
        )
        return self._config_params.get("algorithm", None)
