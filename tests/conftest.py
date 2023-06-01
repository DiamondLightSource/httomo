# Defines common fixtures and makes them available to all tests

import os
import sys
from pathlib import Path
from shutil import rmtree

import pytest
import yaml


def pytest_configure(config):
    config.addinivalue_line("markers", "mpi: mark test to run in an MPI environment")
    config.addinivalue_line("markers", "perf: mark test as performance test")
    config.addinivalue_line("markers", "cupy: needs cupy to run")


def pytest_addoption(parser):
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests only",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--performance"):
        skip_other = pytest.mark.skip(reason="not a performance test")
        for item in items:
            if "perf" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="performance test - use '--performance' to run"
        )
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)


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
        "-m",
        "httomo",
        "run",
        "--save_all",
        "--ncore",
        "2",
        "output_dir/",
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
def i12_data():
    return "tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"


@pytest.fixture
def i12_loader():
    return "samples/pipeline_template_examples/DLS/03_i12_separate_darks_flats.yaml"


@pytest.fixture
def standard_loader():
    return "samples/loader_configs/standard_tomo.yaml"


@pytest.fixture
def more_than_one_method():
    return "samples/pipeline_template_examples/testing/more_than_one_method.yaml"


@pytest.fixture
def sample_pipelines():
    return "samples/pipeline_template_examples/"


@pytest.fixture
def gpu_pipeline():
    return "samples/pipeline_template_examples/04_basic_gpu_pipeline_tomo_standard.yaml"


@pytest.fixture
def merge_yamls():
    def _merge_yamls(*yamls) -> None:
        """Merge multiple yaml files into one"""
        data = []
        for y in yamls:
            with open(y, "r") as file_descriptor:
                data.extend(yaml.load(file_descriptor, Loader=yaml.SafeLoader))
        with open("temp.yaml", "w") as file_descriptor:
            yaml.dump(data, file_descriptor)

    return _merge_yamls
