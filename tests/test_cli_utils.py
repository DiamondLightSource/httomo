from pathlib import Path
import pytest

from httomo.cli_utils import is_sweep_pipeline


@pytest.mark.parametrize(
    "pipeline_file, expected_is_sweep_pipeline",
    [
        ("samples/pipeline_template_examples/testing/sweep_range.yaml", True),
        ("samples/pipeline_template_examples/testing/example.yaml", False),
    ],
)
def test_is_sweep_pipeline(pipeline_file: Path, expected_is_sweep_pipeline: bool):
    pipeline_file_path = Path(__file__).parent / pipeline_file
    assert is_sweep_pipeline(pipeline_file_path) is expected_is_sweep_pipeline
