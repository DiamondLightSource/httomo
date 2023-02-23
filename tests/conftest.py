# Defines common fixtures and makes them available to all tests

import os
import sys
from pathlib import Path
from shutil import rmtree

import pytest
import yaml


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
def merge_yamls():
    def _merge_yamls(*yamls) -> None:
        '''Merge multiple yaml files into one'''
        data = []
        for y in yamls:
            with open(y, "r") as file_descriptor:
                data.extend(yaml.load(file_descriptor, Loader=yaml.SafeLoader))
        with open("temp.yaml", "w") as file_descriptor:
            yaml.dump(data, file_descriptor)

    return _merge_yamls
