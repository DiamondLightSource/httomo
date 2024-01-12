"""
Some unit tests for the yaml checker
"""
from typing import Callable

import pytest
import yaml

from httomo.yaml_checker import (
    check_all_stages_defined,
    check_all_stages_non_empty,
    check_hdf5_paths_against_loader,
    check_loading_stage_one_method,
    check_methods_exist_in_templates,
    check_one_method_per_module,
    check_valid_method_parameters,
    sanity_check,
    validate_yaml_config,
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


def test_missing_loader_stage(
        sample_pipelines: str,
        load_yaml: Callable
):
    missing_loader_stage_pipeline = (
        sample_pipelines + "testing/missing_loader_stage.yaml"
    )
    conf = load_yaml(missing_loader_stage_pipeline)
    assert not check_all_stages_defined(conf)


def test_empty_loader_stage(
        sample_pipelines: str,
        load_yaml: Callable
):
    empty_loader_stage_pipeline = (
        sample_pipelines + "testing/empty_loader_stage.yaml"
    )
    conf = load_yaml(empty_loader_stage_pipeline)
    assert not check_all_stages_non_empty(conf)


def test_invalid_loader_stage(
        sample_pipelines,
        load_yaml: Callable
):
    invalid_loader_stage_pipeline = (
        sample_pipelines + "testing/invalid_loader_stage.yaml"
    )
    conf = load_yaml(invalid_loader_stage_pipeline)
    assert not check_loading_stage_one_method(conf)


def test_one_method_per_module(
        more_than_one_method,
        load_yaml: Callable
):
    conf = load_yaml(more_than_one_method)
    assert not check_one_method_per_module(conf)


def test_hdf5_paths_against_loader(
        standard_data,
        sample_pipelines,
        load_yaml: Callable
):
    incorrect_path_pipeline = (
        sample_pipelines + "testing/incorrect_path.yaml"
    )
    conf = load_yaml(incorrect_path_pipeline)
    assert not check_hdf5_paths_against_loader(conf[0][0], standard_data)


def test_check_methods_exist_in_templates(
        sample_pipelines: str,
        load_yaml: Callable
):
    incorrect_method_pipeline = (
        sample_pipelines + "testing/incorrect_method.yaml"
    )
    conf = load_yaml(incorrect_method_pipeline)
    assert not check_methods_exist_in_templates(conf)


def test_check_valid_method_parameters(
        sample_pipelines: str,
        load_yaml: Callable
):
    required_param_pipeline = (
        sample_pipelines + "testing/required_param.yaml"
    )
    conf = load_yaml(required_param_pipeline)
    assert not check_valid_method_parameters(conf)


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("pipeline_cpu1.yaml", True),
        ("pipeline_cpu2.yaml", True),
        ("pipeline_gpu1.yaml", True),
        #("parameter_sweeps/02_median_filter_kernel_sweep.yaml", True),
    ],
    ids=[
        "cpu1_pipeline",
        "cpu2_pipeline",
        "gpu1_pipeline",
    ],
)
def test_validate_yaml_config(
    sample_pipelines: str,
    yaml_file: str,
    standard_data: str,
    expected: bool
):
    yaml_file = sample_pipelines + yaml_file
    assert validate_yaml_config(yaml_file, standard_data) == expected


