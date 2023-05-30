"""
Some unit tests for the yaml checker
"""
import pytest

from httomo.yaml_checker import (
    check_one_method_per_module,
    sanity_check,
    validate_yaml_config,
)


def test_sanity_check(sample_pipelines):
    wrong_indentation_pipeline = (
        sample_pipelines + "testing/wrong_indentation_pipeline.yaml"
    )
    assert not sanity_check(wrong_indentation_pipeline)


def test_one_method_per_module(more_than_one_method):
    assert not check_one_method_per_module(more_than_one_method)


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("testing/testing_pipeline.yaml", False),
        ("testing/incorrect_method.yaml", False),
        ("02_basic_cpu_pipeline_tomo_standard.yaml", True),
        ("04_basic_gpu_pipeline_tomo_standard.yaml", True),
        ("multi_inputs/01_dezing_multi_inputs.yaml", True),
        ("parameter_sweeps/02_median_filter_kernel_sweep.yaml", True),
        ("testing/incorrect_path.yaml", False),
        ("testing/required_param.yaml", False),
    ],
    ids=[
        "no_loader_pipeline",
        "incorrect_method",
        "cpu_pipeline",
        "gpu_pipeline",
        "multi_input_pipeline",
        "sweep_pipeline",
        "incorrect_path",
        "required_param",
    ],
)
def test_validate_yaml_config(sample_pipelines, yaml_file, standard_data, expected):
    yaml_file = sample_pipelines + yaml_file
    assert validate_yaml_config(yaml_file, standard_data) == expected
