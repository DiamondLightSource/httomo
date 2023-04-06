from typing import List, Union
import yaml
from pathlib import Path


YAML_DIR = Path(__file__).parent / "packages/"


def get_httomolib_method_meta(method_path: Union[List[str], str]):
    """
    Get full method meta information for a httomolib method.

    Parameters
    ----------
    method_path : List[str] | str
        Path to the method, either as ["prep", "normalize", "normalize"] or "prep.normalize.normalize"

    Returns
    -------
    httomolib.MethodMeta
        Full method meta information as exported from httomolib
    """
    if isinstance(method_path, str):
        method_path = method_path.split(".")

    from httomolib import method_registry, MethodMeta

    info = method_registry["httomolib"]
    for key in method_path:
        try:
            info = info[key]
        except KeyError:
            raise KeyError(
                f"Method {'.'.join(method_path)} not found in httomolib registry"
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
        name. Ie, `httomolib.misc.images.save_to_images`.

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
    if package_name == "httomolib":
        return _get_method_info_httomolib(split_method_path[1:], attr)

    yaml_info_path = Path(YAML_DIR, f"{package_name}.yaml")

    if not yaml_info_path.exists():
        err_str = f"The YAML file {yaml_info_path} doesn't exist."
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


def _get_method_info_httomolib(method_path: List[str], attr: str):
    meta = get_httomolib_method_meta(method_path)

    try:
        return getattr(meta, attr)
    except KeyError:
        raise KeyError(
            f"The attribute {attr} is not present on httomolib.{'.'.join(method_path)}"
        )
