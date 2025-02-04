# Defines common fixtures and makes them available to all tests

import os
import sys
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable, Dict, List, TypeAlias
import numpy as np

import pytest
import yaml
from httomo.darks_flats import DarksFlatsFileConfig
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock

MethodConfig: TypeAlias = Dict[str, Any]
PipelineConfig: TypeAlias = List[MethodConfig]


CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "mpi: mark test to run in an MPI environment")
    config.addinivalue_line("markers", "perf: mark test as performance test")
    config.addinivalue_line("markers", "cupy: needs cupy to run")
    config.addinivalue_line(
        "markers", "pipesmall: mark tests to run full pipelines on small data"
    )
    config.addinivalue_line(
        "markers", "pipebig: mark tests to run full pipelines on raw big data"
    )
    config.addinivalue_line(
        "markers", "preview: mark test to run with `httomo preview`"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests only",
    )
    parser.addoption(
        "--pipeline_small",
        action="store_true",
        default=False,
        help="run full pipelines on small data",
    )
    parser.addoption(
        "--pipeline_big",
        action="store_true",
        default=False,
        help="run full pipelines on raw (big) data",
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
    if config.getoption("--pipeline_small"):
        skip_other = pytest.mark.skip(reason="not a pipeline small data test")
        for item in items:
            if "pipesmall" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="pipeline small data test - use '--pipeline_small' to run"
        )
        for item in items:
            if "pipesmall" in item.keywords:
                item.add_marker(skip_perf)
    if config.getoption("--pipeline_big"):
        skip_other = pytest.mark.skip(reason="not a pipeline raw big data test")
        for item in items:
            if "pipebig" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="pipeline raw big data test - use '--pipeline_big' to run"
        )
        for item in items:
            if "pipebig" in item.keywords:
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


@pytest.fixture
def data360():
    return "tests/test_data/360scan/360scan.hdf"


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
def data(host_data, ensure_clean_memory):
    import cupy as cp

    return cp.asarray(host_data)


@pytest.fixture
def host_angles(data_file):
    return np.float32(np.copy(data_file["angles"]))


@pytest.fixture
def host_angles_radians(host_angles):
    return host_angles


@pytest.fixture
def host_flats(data_file):
    return np.float32(np.copy(data_file["flats"]))


@pytest.fixture
def flats(host_flats, ensure_clean_memory):
    import cupy as cp

    return cp.asarray(host_flats)


@pytest.fixture
def host_darks(
    data_file,
):
    return np.float32(np.copy(data_file["darks"]))


@pytest.fixture
def darks(host_darks, ensure_clean_memory):
    import cupy as cp

    return cp.asarray(host_darks)


@pytest.fixture
def standard_data_path():
    return "/entry1/tomo_entry/data/data"


@pytest.fixture
def standard_image_key_path():
    return "/entry1/tomo_entry/instrument/detector/image_key"


# TODO: depricate when loader is generalised (big data tests instead)
@pytest.fixture
def diad_data():
    return "tests/test_data/k11_diad/k11-18014.nxs"


# TODO: depricate when loader is generalised
@pytest.fixture
def diad_loader():
    return "tests/samples/loader_configs/diad.yaml"


@pytest.fixture
def standard_loader():
    return "tests/samples/loader_configs/standard_tomo.yaml"


@pytest.fixture
def sample_pipelines():
    return "tests/samples/pipeline_template_examples/"


###########Auto-generated pipelines##################


@pytest.fixture
def cpu_pipeline_gridrec():
    return "docs/source/pipelines_full/cpu_pipeline_gridrec.yaml"


@pytest.fixture
def gpu_pipelineFBP():
    return "docs/source/pipelines_full/gpu_pipelineFBP.yaml"


@pytest.fixture
def gpu_pipelineFBP_denoising():
    return "docs/source/pipelines_full/gpu_pipelineFBP_denoising.yaml"


@pytest.fixture
def gpu_pipeline_diad_FBP_noimagesaving():
    return "docs/source/pipelines_full/gpu_diad_FBP_noimagesaving.yaml"


@pytest.fixture
def gpu_pipeline_diad_FBP():
    return "docs/source/pipelines_full/gpu_diad_FBP.yaml"


# ---------------------END------------------------#

###########Raw projection data (big)##################
# Note that raw_data folder should exist with the datasets
# listed bellow


@pytest.fixture
def diad_k11_38727():
    # 4k projections, 45gb dataset
    return "tests/test_data/raw_data/diad/k11-38727.nxs"


@pytest.fixture
def diad_k11_38729():
    # 2k projections, 22gb dataset
    return "tests/test_data/raw_data/diad/k11-38729.nxs"


@pytest.fixture
def diad_k11_38730():
    # 1k projections, 11gb dataset
    return "tests/test_data/raw_data/diad/k11-38730.nxs"


@pytest.fixture
def diad_k11_38731():
    # 0.5k projections, 6gb dataset
    return "tests/test_data/raw_data/diad/k11-38731.nxs"


@pytest.fixture
def gpu_diad_FBP_k11_38731_npz():
    # 10 slices numpy array
    return np.load("tests/test_data/raw_data/diad/gpu_diad_FBP_k11-38731.npz")


# ---------------------END------------------------#


# TODO: depricate when loader is generalised
@pytest.fixture
def diad_pipeline_gpu():
    return "tests/samples/pipeline_template_examples/DLS/01_diad_pipeline_gpu.yaml"


@pytest.fixture
def i12_data():
    return "tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"


# TODO: move to big pipeline tests
@pytest.fixture
def pipeline360():
    return "samples/pipeline_template_examples/DLS/02_i12_360scan_pipeline.yaml"


@pytest.fixture
def i12_loader():
    return (
        "tests/samples/pipeline_template_examples/DLS/03_i12_separate_darks_flats.yaml"
    )


@pytest.fixture
def i12_loader_ignore_darks_flats():
    return "tests/samples/pipeline_template_examples/DLS/04_i12_ignore_darks_flats.yaml"


@pytest.fixture
def yaml_gpu_pipeline360_2():
    return "tests/samples/pipeline_template_examples/pipeline_360deg_gpu2.yaml"


###########Sweep pipelines (not autogenerated currently)###############
@pytest.fixture
def yaml_gpu_pipeline_sweep_cor():
    return "tests/samples/pipeline_template_examples/parameter-sweep-cor.yaml"


@pytest.fixture
def yaml_gpu_pipeline_sweep_paganin():
    return "tests/samples/pipeline_template_examples/parameter-sweep-paganin.yaml"


# ---------------------END------------------------#


@pytest.fixture(scope="session")
def distortion_correction_path(test_data_path):
    return os.path.join(test_data_path, "distortion-correction")


@pytest.fixture
def merge_yamls(load_yaml: Callable):
    def _merge_yamls(*yamls) -> None:
        """Merge multiple yaml files into one"""
        data: List = []
        for y in yamls:
            curr_yaml_list = load_yaml(y)
            for x in curr_yaml_list:
                data.append(x)
        with open("temp.yaml", "w") as file_descriptor:
            yaml.dump(data, file_descriptor)

    return _merge_yamls


@pytest.fixture
def standard_data_darks_flats_config() -> DarksFlatsFileConfig:
    return DarksFlatsFileConfig(
        file=Path(__file__).parent / "test_data/tomo_standard.nxs",
        data_path="/entry1/tomo_entry/data/data",
        image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
    )


@pytest.fixture
def dummy_block() -> DataSetBlock:
    data = np.ones((10, 10, 10), dtype=np.float32)
    aux_data = AuxiliaryData(
        angles=np.ones(data.shape[0], dtype=np.float32),
        darks=2.0 * np.ones((2, data.shape[1], data.shape[2]), dtype=np.float32),
        flats=1.0 * np.ones((2, data.shape[1], data.shape[2]), dtype=np.float32),
    )
    return DataSetBlock(data=data, aux_data=aux_data)


@pytest.fixture()
def get_files():
    def _get_files(dir_path: str, excl: List[str] = []) -> List[str]:
        """Returns list of files from provided directory

        Parameters
        ----------
        dir_path
            Directory to search
        excl
            Exclude files with a path containing any str in this list

        Returns
        -------
        List of file paths
        """
        _dir = Path(dir_path).glob("**/*")
        _files = [
            str(f) for f in _dir if f.is_file() and not any(st in str(f) for st in excl)
        ]
        return _files

    return _get_files


@pytest.fixture()
def load_yaml():
    def _load_yaml(yaml_in: str) -> PipelineConfig:
        """Loads provided yaml and returns dict

        Parameters
        ----------
        yaml_in
            yaml to load

        Returns
        -------
        PipelineConfig
        """
        with open(yaml_in, "r") as f:
            conf = list(yaml.load_all(f, Loader=yaml.FullLoader))
        return conf[0]

    return _load_yaml


def _change_value_parameters_method_pipeline(
    yaml_path: str,
    method: list,
    key: list,
    value: list,
):
    # changes methods parameters in the given pipeline and re-save the pipeline
    with open(yaml_path, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml.FullLoader))
    opened_yaml = conf[0]
    methods_no = len(opened_yaml)
    methods_no_correct = len(method)
    for i in range(methods_no):
        method_content = opened_yaml[i]
        method_name = method_content["method"]
        for j in range(methods_no_correct):
            if method[j] == method_name:
                # change something in parameters here
                opened_yaml[i]["parameters"][key[j]] = value[j]

    with open(yaml_path, "w") as file_descriptor:
        yaml.dump(opened_yaml, file_descriptor)

    return 0
