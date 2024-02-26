from pathlib import Path

import yaml


def get_packages_current_version(package: str, type_p: str = "external") -> str:
    """
    Get the current version of the package.
    'external' points to  httomo/methods_database/packages/external/versions.yaml
    'httomo' points to httomo/methods_database/packages/external/versions.yaml
    """
    if type_p == "external":
        type_p = "methods_database/packages/external/versions.yaml"
    elif type_p == "httomo":
        type_p = "methods_database/packages/version.yaml"
    else:
        raise ValueError(
            "The accepted type of packages is only 'external' and 'httomo'"
        )

    versions_file = Path(__file__).parent / type_p
    with open(versions_file, "r") as f:
        versions = yaml.safe_load(f)

    return str(versions[package]["current"][0])
