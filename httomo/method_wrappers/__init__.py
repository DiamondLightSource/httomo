from httomo.preview import PreviewConfig
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository

from .generic import GenericMethodWrapper

# import all other wrappers to make sure they are available to the factory function
# (add imports here when createing new wrappers)
import httomo.method_wrappers.datareducer
import httomo.method_wrappers.dezinging
import httomo.method_wrappers.distortion_correction
import httomo.method_wrappers.images
import httomo.method_wrappers.reconstruction
import httomo.method_wrappers.rotation
import httomo.method_wrappers.stats_calc
import httomo.method_wrappers.save_intermediate

from mpi4py.MPI import Comm
from typing import Dict, Optional


def make_method_wrapper(
    method_repository: MethodRepository,
    module_path: str,
    method_name: str,
    comm: Comm,
    preview_config: PreviewConfig,
    save_result: Optional[bool] = None,
    output_mapping: Dict[str, str] = {},
    **kwargs,
) -> MethodWrapper:
    """Factory function to generate the appropriate wrapper based on the module
    path and method name. Clients do not need to be concerned about which particular
    derived class is returned.

    Parameters
    ----------

    method_repository: MethodRepository
        Repository of methods that we can use the query properties
    module_path: str
        Path to the module where the method is in python notation, e.g. "httomolibgpu.prep.normalize"
    method_name: str
        Name of the method (function within the given module)
    preview_config : PreviewConfig
        Config for preview value from loader
    comm: Comm
        MPI communicator object
    save_result: Optional[bool]
            Should the method's result be saved to an intermediate h5 file? If not given (or None),
            it queries the method database for the default value.
    output_mapping: Dict[str, str]
        A dictionary mapping output names to translated ones. The side outputs will be renamed
        as specified, if the parameter is given. If not, no side outputs will be passed on.
    kwargs:
        Arbitrary keyword arguments that get passed to the method as parameters.

    Returns
    -------

    MethodWrapper
        An instance of a wrapper class
    """

    # go throw all subclasses of GenericMethodWrapper and see which one should be instantiated,
    # based on module path and method name
    cls = GenericMethodWrapper
    for c in GenericMethodWrapper.__subclasses__():
        if c.should_select_this_class(module_path, method_name):
            assert cls == GenericMethodWrapper, (
                f"The values returned from should_select_this_class('{module_path}', '{method_name}')"
                + f" are ambigious between {c.__name__} and {cls.__name__}"
            )
            cls = c

    if cls.requires_preview():
        return cls(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            preview_config=preview_config,
            save_result=save_result,
            output_mapping=output_mapping,
            **kwargs,
        )
    else:
        return cls(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            save_result=save_result,
            output_mapping=output_mapping,
            **kwargs,
        )
