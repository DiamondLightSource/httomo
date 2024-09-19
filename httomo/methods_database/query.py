from importlib import import_module
from types import ModuleType
from typing import Callable, List, Literal, Optional, Tuple
from pathlib import Path
import numpy as np

import yaml
from httomo.runner.methods_repository_interface import GpuMemoryRequirement, MethodQuery

from httomo.utils import Pattern, log_exception
from httomo.runner.methods_repository_interface import MethodRepository

YAML_DIR = Path(__file__).parent / "packages/"


def get_method_info(module_path: str, method_name: str, attr: str):
    """Get the information about the given method associated with `attr` that
    is stored in the relevant YAML file in `httomo/methods_database/packages/`

    Parameters
    ----------
    module_path : str
        The full module path of the method, including the top-level package
        name. Ie, `httomolib.misc.images.save_to_images`.

    method_name : str
        The name of the method function.

    attr : str
        The name of the piece of information about the method being requested
        (for example, "pattern").

    Returns
    -------
    The requested piece of information about the method.
    """
    method_path = f"{module_path}.{method_name}"
    split_method_path = method_path.split(".")
    package_name = split_method_path[0]

    # open the library file for the package
    ext_package_path = ""
    if package_name != "httomo":
        ext_package_path = f"external/{package_name}/"
    yaml_info_path = Path(YAML_DIR, str(ext_package_path), f"{package_name}.yaml")
    if not yaml_info_path.exists():
        err_str = f"The YAML file {yaml_info_path} doesn't exist."
        log_exception(err_str)
        raise FileNotFoundError(err_str)

    with open(yaml_info_path, "r") as f:
        info = yaml.safe_load(f)
        for key in split_method_path[1:]:
            try:
                info = info[key]
            except KeyError:
                raise KeyError(f"The key {key} is not present ({method_path})")

    try:
        return info[attr]
    except KeyError:
        raise KeyError(f"The attribute {attr} is not present on {method_path}")


# Implementation of methods database query class
class MethodsDatabaseQuery(MethodQuery):
    def __init__(self, module_path: str, method_name: str):
        self.module_path = module_path
        self.method_name = method_name

    def get_pattern(self) -> Pattern:
        p = get_method_info(self.module_path, self.method_name, "pattern")
        assert p in ["projection", "sinogram", "all"], (
            f"The pattern {p} that is listed for the method "
            f"{self.module_path}.{self.method_name} is invalid."
        )
        if p == "projection":
            return Pattern.projection
        if p == "sinogram":
            return Pattern.sinogram
        return Pattern.all

    def get_output_dims_change(self) -> bool:
        p = get_method_info(self.module_path, self.method_name, "output_dims_change")
        return bool(p)

    def get_implementation(self) -> Literal["cpu", "gpu", "gpu_cupy"]:
        p = get_method_info(self.module_path, self.method_name, "implementation")
        assert p in [
            "gpu",
            "gpu_cupy",
            "cpu",
        ], f"The implementation arch {p} listed for method {self.module_path}.{self.method_name} is invalid"
        return p

    def save_result_default(self) -> bool:
        return get_method_info(
            self.module_path, self.method_name, "save_result_default"
        )

    def padding(self) -> bool:
        return get_method_info(self.module_path, self.method_name, "padding")

    def get_memory_gpu_params(
        self,
    ) -> Optional[GpuMemoryRequirement]:
        p = get_method_info(self.module_path, self.method_name, "memory_gpu")
        if p is None or p == "None":
            return None
        if type(p) == list:
            # convert to dict first
            d: dict = dict()
            for item in p:
                d |= item
        else:
            d = p
        return GpuMemoryRequirement(multiplier=d["multiplier"], method=d["method"])

    def calculate_memory_bytes(
        self, non_slice_dims_shape: Tuple[int, int], dtype: np.dtype, **kwargs
    ) -> Tuple[int, int]:
        smodule = self._import_supporting_funcs_module()
        module_mem: Callable = getattr(
            smodule, "_calc_memory_bytes_" + self.method_name
        )
        memory_bytes: Tuple[int, int] = module_mem(
            non_slice_dims_shape, dtype, **kwargs
        )
        return memory_bytes

    def calculate_output_dims(
        self, non_slice_dims_shape: Tuple[int, int], **kwargs
    ) -> Tuple[int, int]:
        smodule = self._import_supporting_funcs_module()
        module_mem: Callable = getattr(smodule, "_calc_output_dim_" + self.method_name)
        return module_mem(non_slice_dims_shape, **kwargs)

    def calculate_padding(self, **kwargs) -> Tuple[int, int]:
        smodule = self._import_supporting_funcs_module()
        module_pad: Callable = getattr(smodule, "_calc_padding_" + self.method_name)
        return module_pad(**kwargs)

    def _import_supporting_funcs_module(self) -> ModuleType:

        module_mem_path = "httomo.methods_database.packages.external."
        path = self.module_path.split(".")
        path.insert(1, "supporting_funcs")
        module_mem_path += ".".join(path)
        return import_module(module_mem_path)

    def swap_dims_on_output(self) -> bool:
        return self.module_path.startswith("tomopy.recon")


class MethodDatabaseRepository(MethodRepository):
    def query(self, module_path: str, method_name: str) -> MethodQuery:
        return MethodsDatabaseQuery(module_path, method_name)
