import yaml
from typing import List, Dict
from pathlib import Path


def open_yaml_config(filepath:Path) -> List[Dict]:
    """Open and read a given YAML config file into a python data structure

    Args:
        filepath: The file containing the YAML config
    """

    with open(filepath, 'r') as f:
        conf = yaml.safe_load(f)

    # TODO: validate the YAML to ensure there are no missing fields that are
    # required

    # TODO: perform parameter type checks

    return conf


def validate_yaml_config() -> bool:
    """Check that the pipeline YAML config isn't missing any required fields

    Args:
    """
    pass


def check_param_types() -> bool:
    """Perform type-checking of the values provided in the pipeline YAML config

    Args:
    """
    pass


def generate_conf_from_prerun() -> None:
    """Create a YAML config file based on a pre-run of given data

    Args:
    """
    pass
