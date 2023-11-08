from types import ModuleType
from typing import Callable, List, Literal, Tuple
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

    yaml_info_path = Path(YAML_DIR, f"{package_name}.yaml")

    # get information about the currently supported version of the package
    yaml_versions_path = Path(YAML_DIR, "external/", "versions.yaml")

    if not yaml_versions_path.exists():
        err_str = f"The YAML file {yaml_versions_path} doesn't exist."
        log_exception(err_str)
        raise ValueError(err_str)

    with open(yaml_versions_path, "r") as f:
        yaml_versions_library = yaml.safe_load(f)

    ext_package_path = ""
    for module, versions_dict in yaml_versions_library.items():
        if module == package_name:
            for version_type, package_version in versions_dict.items():
                if version_type == "current":
                    package_version = package_version[0]
                    ext_package_path = f"external/{package_name}/{package_version}/"

    # open the library file for the package
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
        if p == "projection":
            return Pattern.projection
        if p == "sinogram":
            return Pattern.sinogram
        if p == "all":
            return Pattern.all
        raise ValueError(
            f"The pattern {p} that is listed for the method "
            f"{self.module_path}.{self.method_name} is invalid."
        )

    def get_output_dims_change(self) -> bool:
        p = get_method_info(self.module_path, self.method_name, "output_dims_change")
        return bool(p)

    def get_implementation(self) -> Literal["cpu", "gpu", "gpu_cupy"]:
        p = get_method_info(self.module_path, self.method_name, "implementation")
        if p not in ["gpu", "gpu_cupy", "gpu"]:
            raise ValueError(
                f"The ipmlementation arch {p} listed for method {self.module_path}.{self.method_name} is invalid"
            )
        return p

    def get_memory_gpu_params(
        self,
    ) -> List[GpuMemoryRequirement]:
        p = get_method_info(self.module_path, self.method_name, "memory_gpu")
        if p is None or p == "None":
            return []
        if type(p) == list:
            # convert to dict first
            dd = dict()
            for item in p:
                dd |= item
        else:
            dd = p
        # now iterate and make it into one
        assert (
            len(dd["datasets"]) == len(dd["multipliers"]) == len(dd["methods"])
        ), "Invalid data"
        return [
            GpuMemoryRequirement(
                dataset=d, multiplier=dd["multipliers"][i], method=dd["methods"][i]
            )
            for i, d in enumerate(dd["datasets"])
        ]

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

    def _import_supporting_funcs_module(self) -> ModuleType:
        from importlib import import_module

        module_mem_path = "httomo.methods_database.packages.external."
        path = self.module_path.split(".")
        path.insert(1, "supporting_funcs")
        module_mem_path += ".".join(path)
        return import_module(module_mem_path)


class MethodDatabaseRepository(MethodRepository):
    def query(self, module_path: str, method_name: str) -> MethodQuery:
        return MethodsDatabaseQuery(module_path, method_name)
