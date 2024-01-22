"""
Some unit tests for the yaml checker
"""
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


# TODO: yaml checker needs to be modified first 
"""
def test_sanity_check(sample_pipelines, yaml_loader: type[YamlLoader]):
    wrong_indentation_pipeline = (
        sample_pipelines + "testing/wrong_indentation_pipeline.yaml"
    )
    with open(wrong_indentation_pipeline, "r") as f:
        conf_generator = yaml.load_all(f, Loader=yaml_loader)
        # `assert` needs to be in `with` block for this case, because
        # `conf_generator` is lazy-loaded from the file when converted to a
        # list inside `sanity_check()`
        assert not sanity_check(conf_generator)


def test_missing_loader_stage(sample_pipelines, yaml_loader: type[YamlLoader]):
    missing_loader_stage_pipeline = (
        sample_pipelines + "testing/missing_loader_stage.yaml"
    )
    with open(missing_loader_stage_pipeline, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml_loader))
    assert not check_all_stages_defined(conf)


def test_empty_loader_stage(sample_pipelines, yaml_loader: type[YamlLoader]):
    empty_loader_stage_pipeline = (
        sample_pipelines + "testing/empty_loader_stage.yaml"
    )
    with open(empty_loader_stage_pipeline, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml_loader))
    assert not check_all_stages_non_empty(conf)


def test_invalid_loader_stage(sample_pipelines, yaml_loader: type[YamlLoader]):
    invalid_loader_stage_pipeline = (
        sample_pipelines + "testing/invalid_loader_stage.yaml"
    )
    with open(invalid_loader_stage_pipeline, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml_loader))
    assert not check_loading_stage_one_method(conf)


def test_one_method_per_module(more_than_one_method, yaml_loader: type[YamlLoader]):
    with open(more_than_one_method, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml_loader))
    assert not check_one_method_per_module(conf)


def test_hdf5_paths_against_loader(
        standard_data,
        sample_pipelines,
        yaml_loader: type[YamlLoader]
):
    incorrect_path_pipeline = (
        sample_pipelines + "testing/incorrect_path.yaml"
    )
    with open(incorrect_path_pipeline, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml_loader))
    assert not check_hdf5_paths_against_loader(conf[0][0], standard_data)


def test_check_methods_exist_in_templates(
        sample_pipelines,
        yaml_loader: type[YamlLoader]
):
    incorrect_method_pipeline = (
        sample_pipelines + "testing/incorrect_method.yaml"
    )
    with open(incorrect_method_pipeline, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml_loader))
    assert not check_methods_exist_in_templates(conf)


def test_check_valid_method_parameters(
        sample_pipelines,
        yaml_loader: type[YamlLoader]
):
    required_param_pipeline = (
        sample_pipelines + "testing/required_param.yaml"
    )
    with open(required_param_pipeline, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml_loader))
    assert not check_valid_method_parameters(conf, yaml_loader)


@pytest.mark.parametrize(
    "yaml_file, expected",
    [
        ("02_basic_cpu_pipeline_tomo_standard.yaml", True),
        ("03_basic_gpu_pipeline_tomo_standard.yaml", True),
        ("parameter_sweeps/02_median_filter_kernel_sweep.yaml", True),
    ],
    ids=[
        "cpu_pipeline",
        "gpu_pipeline",
        "sweep_pipeline",
    ],
)
def test_validate_yaml_config(
    sample_pipelines: str,
    yaml_file: str,
    standard_data: str,
    expected: bool,
    yaml_loader: type[YamlLoader]
):
    yaml_file = sample_pipelines + yaml_file
    assert validate_yaml_config(yaml_file, yaml_loader, standard_data) == expected

"""
