"""
Module for checking the validity of yaml files.
"""
import os
import yaml
from httomo.yaml_utils import open_yaml_config

__all__ = [
    "check_one_method_per_module",
    "sanity_check",
    "validate_yaml_config",
]


def sanity_check(yaml_file):
    """
    Check if the yaml file is properly indented, has valid mapping and tags.
    """
    with open(yaml_file, "r") as file:
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
            print(
                f"Invalid tag in the YAML_CONFIG file at line {e.problem_mark.line + 1}."
            )
        except yaml.reader.ReaderError as e:
            print(f"Failed to parse YAML file at line {e.problem_mark.line + 1}: {e}")
        except yaml.YAMLError as e:
            if hasattr(e, "problem_mark"):
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


def validate_yaml_config(yaml_file):
    """
    Check that the modules, methods, and parameters in the `YAML_CONFIG` file
    are valid, and adhere to the same structure as in each corresponding
    module in `httomo.templates`.
    """
    yaml_data = check_one_method_per_module(yaml_file)

    modules = [next(iter(d)) for d in yaml_data]
    methods = [next(iter(d.values())) for d in yaml_data]
    packages = [m.split(".")[0] for m in modules]

    #: the first method is always a loader
    #: so `testing_pipeline.yaml` should not pass.
    if next(iter(methods[0])) != "standard_tomo":
        print(
            "The first method in the YAML_CONFIG file must be a loader. "
            "Please recheck the yaml file."
        )
        return False

    parent_dir = os.path.dirname(os.path.abspath("__file__"))
    templates_dir = os.path.join(parent_dir, "templates")
    assert os.path.exists(templates_dir)

    _template_yaml_files = [
        os.path.join(
            templates_dir, packages[i], modules[i], next(iter(methods[i])) + ".yaml"
        )
        for i in range(len(modules))
    ]

    for i, f in enumerate(_template_yaml_files):
        if not os.path.exists(f):
            print(
                f"'{modules[i] + '/' + next(iter(methods[i]))}' is not a valid"
                 " path to a method. Please recheck the yaml file."
            )
            return False

    _template_yaml_data_list = [
        next(iter(d.values()))
        for f in _template_yaml_files
        for d in open_yaml_config(f)
    ]

    for i, _ in enumerate(modules):
        d1 = methods[i]
        d2 = _template_yaml_data_list[i]

        for key in d1.keys():
            for parameter in d1[key].keys():
                assert isinstance(parameter, str)
                if parameter not in d2[key].keys():
                    print(
                        f"Parameter '{parameter}' in the '{modules[i]}' method is not valid."
                    )
                    return False

                if None in (d1[key][parameter], d2[key][parameter]):
                    continue

                if not isinstance(d1[key][parameter], type(d2[key][parameter])):
                    print(
                        f"Value assigned to parameter '{parameter}' in the '{modules[i]}' method"
                        f" is not correct. It should be of type {type(d2[key][parameter])}."
                    )
