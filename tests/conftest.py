# Defines common fixtures and makes them available to all tests

import os
import sys
from pathlib import Path
from shutil import rmtree
import numpy as np

import pytest
import yaml
from httomo.runner.dataset import DataSet
from httomo.ui_layer import _yaml_loader


CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def pytest_configure(config):
    config.addinivalue_line("markers", "mpi: mark test to run in an MPI environment")
    config.addinivalue_line("markers", "perf: mark test as performance test")
    config.addinivalue_line("markers", "cupy: needs cupy to run")
    config.addinivalue_line("markers", "preview: mark test to run with `httomo preview`")


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
    return str(Path("output_dir").resolve())


@pytest.fixture
def cmd():
    return [
        sys.executable,
        "-m",
        "httomo",
        "run",
        "--save-all",
    ]


@pytest.fixture
def standard_data():
    return "tests/test_data/tomo_standard.nxs"
    
@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")
    
# only load from disk once per session, and we use np.copy for the elements,
# to ensure data in this loaded file stays as originally loaded
@pytest.fixture(scope="session")
def data_file(test_data_path):
    in_file = os.path.join(test_data_path, "tomo_standard.npz")
    # keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    return np.load(in_file)

@pytest.fixture
@pytest.mark.cupy
def ensure_clean_memory():
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()    


@pytest.fixture
def host_data(data_file):
    return np.float32(np.copy(data_file["data"]))
    
    
@pytest.fixture
@pytest.mark.cupy
def data(host_data, ensure_clean_memory):
    import cupy as cp
    return cp.asarray(host_data)

@pytest.fixture
def host_angles(data_file):
    return np.float32(np.copy(data_file["angles"]))

@pytest.fixture
@pytest.mark.cupy
def angles(host_angles, ensure_clean_memory):
    import cupy as cp
    return cp.asarray(host_angles)

@pytest.fixture
def host_angles_radians(host_angles):
    return host_angles

@pytest.fixture
@pytest.mark.cupy
def angles_radians(angles):
    return angles

@pytest.fixture
def host_flats(data_file):
    return np.float32(np.copy(data_file["flats"]))


@pytest.fixture
@pytest.mark.cupy
def flats(host_flats, ensure_clean_memory):
    import cupy as cp
    return cp.asarray(host_flats)


@pytest.fixture
def host_darks(
    data_file,
):
    return np.float32(np.copy(data_file["darks"]))


@pytest.fixture
@pytest.mark.cupy
def darks(host_darks, ensure_clean_memory):
    import cupy as cp
    return cp.asarray(host_darks)


@pytest.fixture
def standard_data_path():
    return "/entry1/tomo_entry/data/data"


@pytest.fixture
def standard_image_key_path():
    return "/entry1/tomo_entry/instrument/detector/image_key"


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
def i12_loader_ignore_darks_flats():
    return "samples/pipeline_template_examples/DLS/04_i12_ignore_darks_flats.yaml"


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
    return "samples/pipeline_template_examples/03_basic_gpu_pipeline_tomo_standard.yaml"

@pytest.fixture
def python_cpu_pipeline1():
    return "samples/python_templates/pipeline_cpu1.py"

@pytest.fixture
def python_cpu_pipeline2():
    return "samples/python_templates/pipeline_cpu2.py"

@pytest.fixture
def python_cpu_pipeline3():
    return "samples/python_templates/pipeline_cpu3.py"

@pytest.fixture
def python_gpu_pipeline1():
    return "samples/python_templates/pipeline_gpu1.py"

@pytest.fixture
def yaml_cpu_pipeline1():
    return "samples/pipeline_template_examples/pipeline_cpu1.yaml"

@pytest.fixture
def yaml_cpu_pipeline2():
    return "samples/pipeline_template_examples/pipeline_cpu2.yaml"

@pytest.fixture
def yaml_cpu_pipeline3():
    return "samples/pipeline_template_examples/pipeline_cpu3.yaml"

@pytest.fixture
def yaml_gpu_pipeline1():
    return "samples/pipeline_template_examples/pipeline_gpu1.yaml"

@pytest.fixture(scope="session")
def distortion_correction_path(test_data_path):
    return os.path.join(test_data_path, "distortion-correction")

@pytest.fixture
def merge_yamls():
    def _merge_yamls(*yamls) -> None:
        """Merge multiple yaml files into one"""
        data : list = []
        for y in yamls:
            curr_yaml_list = _yaml_loader(y)[0]
            for x in curr_yaml_list:
                data.append(x)
        with open("temp.yaml", "w") as file_descriptor:
            yaml.dump(data, file_descriptor)
    return _merge_yamls

@pytest.fixture
def dummy_dataset() -> DataSet:
    return DataSet(
        data=np.ones((10, 10, 10)),
        angles=np.ones((20,)),
        flats=3 * np.ones((5, 10, 10)),
        darks=2 * np.ones((5, 10, 10)),
    )
