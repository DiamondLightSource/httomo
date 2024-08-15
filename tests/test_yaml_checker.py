"""
Some unit tests for the yaml checker
"""

from typing import Callable
from pathlib import Path

import pytest
import yaml

from httomo.yaml_checker import (
    check_first_method_is_loader,
    check_hdf5_paths_against_loader,
    check_methods_exist_in_templates,
    check_parameter_names_are_known,
    check_parameter_names_are_str,
    check_no_required_parameter_values,
    check_no_duplicated_keys,
    check_ref_id_valid,
    check_id_has_side_out,
    check_side_out_matches_ref_arg,
    check_keys,
    sanity_check,
    validate_yaml_config,
    check_no_imagesaver_after_sweep_method,
)


def test_sanity_check(sample_pipelines):
    wrong_indentation_pipeline = (
        sample_pipelines + "testing/wrong_indentation_pipeline.yaml"
    )
    with open(wrong_indentation_pipeline, "r") as f:
        conf_generator = yaml.load_all(f, Loader=yaml.FullLoader)
        # `assert` needs to be in `with` block for this case, because
        # `conf_generator` is lazy-loaded from the file when converted to a
        # list inside `sanity_check()`
        assert not sanity_check(conf_generator)


def test_check_first_method_is_loader(sample_pipelines: str, load_yaml: Callable):
    no_loader_method_pipeline = sample_pipelines + "testing/no_loader_method.yaml"
    conf = load_yaml(no_loader_method_pipeline)
    assert not check_first_method_is_loader(conf)


def test_hdf5_paths_against_loader(
    standard_data, sample_pipelines, load_yaml: Callable
):
    incorrect_path_pipeline = sample_pipelines + "testing/incorrect_path.yaml"
    conf = load_yaml(incorrect_path_pipeline)
    assert not check_hdf5_paths_against_loader(conf, standard_data)


def test_check_methods_exist_in_templates(sample_pipelines: str, load_yaml: Callable):
    incorrect_method_pipeline = sample_pipelines + "testing/incorrect_method.yaml"
    conf = load_yaml(incorrect_method_pipeline)
    assert not check_methods_exist_in_templates(conf)


@pytest.mark.skip(reason="Some parameters are additional and not listed in templates")
def test_check_parameter_names_are_known(sample_pipelines: str, load_yaml: Callable):
    required_param_pipeline = sample_pipelines + "testing/unknown_param.yaml"
    conf = load_yaml(required_param_pipeline)
    assert not check_parameter_names_are_known(conf)


def test_check_parameter_names_are_str(sample_pipelines: str, load_yaml: Callable):
    required_param_pipeline = sample_pipelines + "testing/non_str_param_name.yaml"
    conf = load_yaml(required_param_pipeline)
    assert not check_parameter_names_are_str(conf)


def test_check_no_required_parameter_values(sample_pipelines: str, load_yaml: Callable):
    required_param_pipeline = sample_pipelines + "testing/required_param.yaml"
    conf = load_yaml(required_param_pipeline)
    assert not check_no_required_parameter_values(conf)


def test_check_no_duplicated_keys(sample_pipelines: str, load_yaml: Callable):
    required_param_pipeline = sample_pipelines + "testing/duplicated_key.yaml"
    assert not check_no_duplicated_keys(Path(required_param_pipeline))


def test_imagesave_after_sweep(sample_pipelines: str, load_yaml: Callable):
    required_param_pipeline = sample_pipelines + "testing/imagesave_after_sweep.yaml"
    assert not check_no_imagesaver_after_sweep_method(Path(required_param_pipeline))


def test_check_keys(sample_pipelines: str, load_yaml: Callable):
    required_param_pipeline = sample_pipelines + "testing/required_keys.yaml"
    conf = load_yaml(required_param_pipeline)
    assert not check_keys(conf)


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("testing/invalid_reference.yaml", True),
        ("testing/invalid_reference_1.yaml", False),
        ("testing/invalid_reference_2.yaml", True),
        ("testing/invalid_reference_3.yaml", True),
        ("testing/valid_reference.yaml", True),
    ],
)
def test_check_id_has_side_out(
    sample_pipelines: str, yaml_file: str, expected: bool, load_yaml: Callable
):
    required_param_pipeline = sample_pipelines + yaml_file
    conf = load_yaml(required_param_pipeline)
    assert check_id_has_side_out(conf) == expected


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("testing/invalid_reference.yaml", True),
        ("testing/invalid_reference_1.yaml", True),
        ("testing/invalid_reference_2.yaml", True),
        ("testing/invalid_reference_3.yaml", False),
        ("testing/valid_reference.yaml", True),
    ],
)
def test_check_ref_id_valid(
    sample_pipelines: str, yaml_file: str, expected: bool, load_yaml: Callable
):
    required_param_pipeline = sample_pipelines + yaml_file
    conf = load_yaml(required_param_pipeline)
    assert check_ref_id_valid(conf) == expected


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("testing/invalid_reference.yaml", False),
        ("testing/invalid_reference_1.yaml", False),
        ("testing/invalid_reference_2.yaml", False),
        ("testing/invalid_reference_3.yaml", False),
        ("testing/valid_reference.yaml", True),
    ],
)
def test_check_side_out_matches_ref_arg(
    sample_pipelines: str, yaml_file: str, expected: bool, load_yaml: Callable
):
    required_param_pipeline = sample_pipelines + yaml_file
    conf = load_yaml(required_param_pipeline)
    assert check_side_out_matches_ref_arg(conf) == expected


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("pipeline_cpu1.yaml", True),
        ("pipeline_cpu2.yaml", True),
        ("pipeline_gpu1.yaml", True),
        ("testing/sweep_manual.yaml", True),
    ],
    ids=[
        "cpu1_pipeline",
        "cpu2_pipeline",
        "gpu1_pipeline",
        "sweep_manual",
    ],
)
def test_validate_yaml_config(
    sample_pipelines: str, yaml_file: str, standard_data: str, expected: bool
):
    yaml_file = sample_pipelines + yaml_file
    assert validate_yaml_config(yaml_file, standard_data) == expected
