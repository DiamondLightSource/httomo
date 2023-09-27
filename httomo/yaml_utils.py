from pathlib import Path

import yaml


def get_external_package_current_version(package: str) -> str:
    """
    Get current version of the external package
    from httomo/methods_database/packages/external/versions.yaml
    """
    versions_file = (
        Path(__file__).parent / "methods_database/packages/external/versions.yaml"
    )
    with open(versions_file, "r") as f:
        versions = yaml.safe_load(f)

    return str(versions[package]["current"][0])

