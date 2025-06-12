from pathlib import Path
import pytest
import json

from httomo.cli_utils import is_sweep_pipeline


@pytest.mark.parametrize(
    "pipeline_file, expected_is_sweep_pipeline",
    [
        ("samples/pipeline_template_examples/testing/sweep_range.yaml", True),
        ("samples/pipeline_template_examples/testing/example.yaml", False),
    ],
)
def test_is_sweep_pipeline_file(pipeline_file: Path, expected_is_sweep_pipeline: bool):
    """Test is_sweep_pipeline with file paths"""
    pipeline_file_path = Path(__file__).parent / pipeline_file
    assert is_sweep_pipeline(pipeline_file_path) is expected_is_sweep_pipeline


def test_is_sweep_pipeline_dict_with_range():
    """Test is_sweep_pipeline with a JSON string containing a sweep range"""
    pipeline_dict = """[
        {
            "method": "normalize",
            "module_path": "httomolibgpu.prep.normalize",
            "parameters": {
                "cutoff": {
                    "start": 10,
                    "stop": 15,
                    "step": 2
                },
                "minus_log": true
            }
        }
    ]"""
    assert is_sweep_pipeline(pipeline_dict) is True


def test_is_sweep_pipeline_dict_with_list():
    """Test is_sweep_pipeline with a JSON string containing a sweep list"""
    pipeline_dict = """[
        {
            "method": "normalize",
            "module_path": "httomolibgpu.prep.normalize",
            "parameters": {
                "cutoff": [5, 10, 15],
                "minus_log": true
            }
        }
    ]"""
    assert is_sweep_pipeline(pipeline_dict) is True
