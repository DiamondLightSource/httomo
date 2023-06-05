from typing import List, Union
from pathlib import Path

import yaml

from httomo.utils import log_exception

YAML_DIR = Path(__file__).parent / "packages/"


def get_httomolibgpu_method_meta(method_path: Union[List[str], str]):
    """
    Get full method meta information for a httomolibgpu method.

    Parameters
    ----------
    method_path : List[str] | str
        Path to the method, either as ["prep", "normalize", "normalize"] or "prep.normalize.normalize"

    Returns
    -------
    httomolibgpu.MethodMeta
        Full method meta information as exported from httomolibgpu
    """
    if isinstance(method_path, str):
        method_path = method_path.split(".")

    from httomolibgpu import method_registry, MethodMeta

    info = method_registry["httomolibgpu"]
    for key in method_path:
        try:
            info = info[key]
        except KeyError:
            raise KeyError(
                f"Method {'.'.join(method_path)} not found in httomolibgpu registry"
            )

    if not isinstance(info, MethodMeta):
        raise ValueError(
            f"method path {'.'.join(method_path)} is not resolving to a method"
        )

    return info


def get_method_info(module_path: str, method_name: str, attr: str):
    """Get the information about the given method associated with `attr` that
    is stored in the relevant YAML file in `httomo/methods_database/packages/`

    Parameters
    ----------
    module_path : str
        The full module path of the method, including the top-level package
        name. Ie, `httomolibgpu.misc.images.save_to_images`.

    method_name : str
        The name of the method function.

    attr : str
        The name of the piece of information about the method being requested
        (for example, "pattern").

    Returns
    -------
    TODO: Needs a "generic" type to represent anything that could be stored in
    the YAML files in the methods database?
        The requested piece of information about the method.
    """
    method_path = f"{module_path}.{method_name}"
    split_method_path = method_path.split(".")
    package_name = split_method_path[0]
    if package_name == "httomolibgpu":
        return _get_method_info_httomolibgpu(split_method_path[1:], attr)

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


def _get_method_info_httomolibgpu(method_path: List[str], attr: str):
    meta = get_httomolibgpu_method_meta(method_path)

    try:
        return getattr(meta, attr)
    except KeyError:
        raise KeyError(
            f"The attribute {attr} is not present on httomolibgpu.{'.'.join(method_path)}"
        )
