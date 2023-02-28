import yaml
from pathlib import Path
from typing import Dict


YAML_DIR = Path(__file__).parent / 'packages/'


def get_method_info(module_path: str, method_name: str) -> Dict:
    """Get the information about the given method that is stored in the relevant
    YAML file in `httomo/methods_database/packages/`

    Parameters
    ----------
    module_path : str
        The full module path of the method, including the top-level package
        name. Ie, `httomolib.misc.images.save_to_images`.

    method_name : str
        The name of the method function.

    Returns
    -------
    Dict
        A dict containing all the relevant information about the method that the
        httomo frameworks needs.
    """
    method_path = f"{module_path}.{method_name}"
    split_method_path = method_path.split('.')
    package_name = split_method_path[0]
    yaml_info_path = Path(YAML_DIR, f"{package_name}.yaml")

    if not yaml_info_path.exists():
        err_str = f"The YAML file {yaml_info_path} doesn't exist."
        raise ValueError(err_str)

    with open(yaml_info_path, 'r') as f:
        info = yaml.safe_load(f)
        for key in split_method_path[1:]:
            info = info[key]

    return info
