"""
Module for checking the validity of yaml files.
"""

import yaml
from httomo.yaml_utils import open_yaml_config

__all__ = ["sanity_check", "check_one_method_per_module"]


def sanity_check(yaml_file):
    """
    Check if the yaml file is properly indented, has valid mapping and tags.
    """
    with open(yaml_file, 'r') as file:
        try:
            yaml_data = open_yaml_config(yaml_file)
        except yaml.parser.ParserError as e:
            line = e.problem_mark.line
            print(
                f"Incorrect indentation in the YAML_CONFIG file at line {line}. "
                "Please recheck the indentation of the file."
            )
        except yaml.scanner.ScannerError as e:
            print(
                f"Incorrect mapping in the YAML_CONFIG file at line {e.problem_mark.line + 1}."
            )
        except yaml.constructor.ConstructorError as e:
            print(f"Invalid tag in the YAML_CONFIG file at line {e.problem_mark.line + 1}.")
        except yaml.reader.ReaderError as e:
            print(f"Failed to parse YAML file at line {e.problem_mark.line + 1}: {e}")
        except yaml.YAMLError as e:
            if hasattr(e, 'problem_mark'):
                print(
                    f"Error in the YAML_CONFIG file at line {e.problem_mark.line}. "
                    "Please recheck the file."
                )

    return yaml_data


def check_one_method_per_module(yaml_file):
    """
    Check that we cannot have a yaml file with more than one method
    being called from one module. For example, we cannot have:

    - tomopy.prep.normalize:
        normalize:
          data_in: tomo
          data_out: tomo
          cutoff: null
        minus_log:
          data_in: tomo
          data_out: tomo
    """
    yaml_data = sanity_check(yaml_file)

    lvalues = [value for d in yaml_data for value in d.values()]
    for i, d in enumerate(lvalues):
        assert isinstance(d, dict)
        if len(d) != 1:
            print(
                f"More than one method is being called from the"
                f" module '{next(iter(yaml_data[i]))}'. "
                "Please recheck the yaml file."
            )

    return yaml_data
