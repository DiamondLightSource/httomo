# Defines common fixtures and makes them available to all tests

from pathlib import Path
from shutil import rmtree

import os
import pytest
import sys


@pytest.fixture
def output_folder():
    if not os.path.exists("output_dir"):
        os.mkdir("output_dir/")
    else:
        for path in Path("output_dir").iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)


@pytest.fixture
def cmd():
    return [
        sys.executable,
        "-m", "httomo",
        "--save_all",
        "--ncore", "2",
        "output_dir/",
        "task_runner",
    ]


@pytest.fixture
def standard_data():
    return "tests/test_data/tomo_standard.nxs"


@pytest.fixture
def testing_pipeline():
    return "samples/pipeline_template_examples/testing/testing_pipeline.yaml"


@pytest.fixture
def diad_data():
    return "tests/test_data/k11_diad/k11-18014.nxs"


@pytest.fixture
def diad_loader():
    return "samples/loader_configs/diad.yaml"


@pytest.fixture
def standard_loader():
    return "samples/loader_configs/standard_tomo.yaml"


@pytest.fixture
def diad_testing_pipeline():
    return "samples/pipeline_template_examples/testing/testing_pipeline_diad.yaml"
