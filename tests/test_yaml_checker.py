"""
Some unit tests for the yaml checker
"""
import pytest

from httomo.yaml_checker import (
    check_all_stages_defined,
    check_all_stages_non_empty,
    check_loading_stage_one_method,
    check_one_method_per_module,
    sanity_check,
    validate_yaml_config,
)
from httomo.yaml_loader import YamlLoader


def test_sanity_check(sample_pipelines):
    wrong_indentation_pipeline = (
        sample_pipelines + "testing/wrong_indentation_pipeline.yaml"
    )
    assert not sanity_check(wrong_indentation_pipeline)


def test_missing_loader_stage(sample_pipelines, yaml_loader: type[YamlLoader]):
    missing_loader_stage_pipeline = (
        sample_pipelines + "testing/missing_loader_stage.yaml"
    )
    assert check_all_stages_defined(missing_loader_stage_pipeline, yaml_loader) is None


def test_empty_loader_stage(sample_pipelines, yaml_loader: type[YamlLoader]):
    empty_loader_stage_pipeline = (
        sample_pipelines + "testing/empty_loader_stage.yaml"
    )
    assert check_all_stages_non_empty(empty_loader_stage_pipeline, yaml_loader) is None


def test_invalid_loader_stage(sample_pipelines, yaml_loader: type[YamlLoader]):
    invalid_loader_stage_pipeline = (
        sample_pipelines + "testing/invalid_loader_stage.yaml"
    )
    assert check_loading_stage_one_method(invalid_loader_stage_pipeline, yaml_loader) is None


def test_one_method_per_module(more_than_one_method):
    assert not check_one_method_per_module(more_than_one_method)


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("testing/testing_pipeline.yaml", False),
        ("testing/incorrect_method.yaml", False),
        ("02_basic_cpu_pipeline_tomo_standard.yaml", True),
        ("03_basic_gpu_pipeline_tomo_standard.yaml", True),
        ("parameter_sweeps/02_median_filter_kernel_sweep.yaml", True),
        ("testing/incorrect_path.yaml", False),
        ("testing/required_param.yaml", False),
    ],
    ids=[
        "no_loader_pipeline",
        "incorrect_method",
        "cpu_pipeline",
        "gpu_pipeline",
        "sweep_pipeline",
        "incorrect_path",
        "required_param",
    ],
)
def test_validate_yaml_config(sample_pipelines, yaml_file, standard_data, expected):
    yaml_file = sample_pipelines + yaml_file
    assert validate_yaml_config(yaml_file, standard_data) == expected
