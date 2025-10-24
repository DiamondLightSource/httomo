# Defines common fixtures and makes them available to all tests

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeAlias, Union, Tuple
import numpy as np
from PIL import Image

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
        "markers", "small_data: mark tests to run full pipelines on small data"
    )
    config.addinivalue_line(
        "markers", "full_data: mark tests to run full pipelines on raw big data"
    )
    config.addinivalue_line(
        "markers",
        "full_data_parallel: mark tests to run full pipelines on raw big data in parallel",
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
        "--small_data",
        action="store_true",
        default=False,
        help="run full pipelines on small data",
    )
    parser.addoption(
        "--full_data",
        action="store_true",
        default=False,
        help="run full pipelines on raw (big) data",
    )
    parser.addoption(
        "--full_data_parallel",
        action="store_true",
        default=False,
        help="run full pipelines on raw (big) data in parallel on two processes",
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
    if config.getoption("--small_data"):
        skip_other = pytest.mark.skip(reason="not a pipeline small data test")
        for item in items:
            if "small_data" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="pipeline small data test - use '--small_data' to run"
        )
        for item in items:
            if "small_data" in item.keywords:
                item.add_marker(skip_perf)
    if config.getoption("--full_data"):
        skip_other = pytest.mark.skip(reason="not a pipeline raw big data test")
        for item in items:
            if "full_data" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="pipeline raw big data test - use '--full_data' to run"
        )
        for item in items:
            if "full_data" in item.keywords:
                item.add_marker(skip_perf)
    if config.getoption("--full_data_parallel"):
        skip_other = pytest.mark.skip(
            reason="not a pipeline raw big data test in parallel"
        )
        for item in items:
            if "full_data_parallel" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="pipeline raw big data test in parallel - use '--full_data_parallel' to run"
        )
        for item in items:
            if "full_data_parallel" in item.keywords:
                item.add_marker(skip_perf)


@pytest.fixture
def output_folder(tmp_path):
    tmp_dir = tmp_path / "output_dir"
    tmp_dir.mkdir()
    return str(Path(tmp_dir).resolve())


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
def cmd_mpirun():
    return [
        "mpirun",
        "-n",
        "2",
        str(sys.executable),
        "-m",
        "httomo",
        "run",
    ]


@pytest.fixture
def standard_data():
    return "tests/test_data/tomo_standard.nxs"


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")


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
def standard_data_path():
    return "/entry1/tomo_entry/data/data"


@pytest.fixture
def standard_image_key_path():
    return "/entry1/tomo_entry/instrument/detector/image_key"


@pytest.fixture
def standard_loader():
    return "tests/samples/loader_configs/standard_tomo.yaml"


@pytest.fixture
def sample_pipelines():
    return "tests/samples/pipeline_template_examples/"


#####################Auto-generated pipelines##################


@pytest.fixture
def FBP3d_tomobar_noimagesaving():
    return "docs/source/pipelines_full/FBP3d_tomobar_noimagesaving.yaml"


@pytest.fixture
def tomopy_gridrec():
    return "docs/source/pipelines_full/tomopy_gridrec.yaml"


@pytest.fixture
def FBP3d_tomobar():
    return "docs/source/pipelines_full/FBP3d_tomobar.yaml"


@pytest.fixture
def LPRec3d_tomobar():
    return "docs/source/pipelines_full/LPRec3d_tomobar.yaml"


@pytest.fixture
def FBP2d_astra():
    return "docs/source/pipelines_full/FBP2d_astra.yaml"


@pytest.fixture
def FBP3d_tomobar_denoising():
    return "docs/source/pipelines_full/FBP3d_tomobar_denoising.yaml"

@pytest.fixture
def FISTA3d_tomobar():
    return "docs/source/pipelines_full/FISTA3d_tomobar.yaml"

@pytest.fixture
def titaren_center_pc_FBP3d_resample():
    return "docs/source/pipelines_full/titaren_center_pc_FBP3d_resample.yaml"


@pytest.fixture
def deg360_paganin_FBP3d_tomobar():
    return "docs/source/pipelines_full/deg360_paganin_FBP3d_tomobar.yaml"


@pytest.fixture
def deg360_distortion_FBP3d_tomobar():
    return "docs/source/pipelines_full/deg360_distortion_FBP3d_tomobar.yaml"


########### Sweep pipelines ###############
@pytest.fixture
def sweep_center_FBP3d_tomobar():
    return "docs/source/pipelines_full/sweep_center_FBP3d_tomobar.yaml"


@pytest.fixture
def sweep_paganin_FBP3d_tomobar():
    return "docs/source/pipelines_full/sweep_paganin_FBP3d_tomobar.yaml"


# ---------------------END------------------------#

###########Raw projection data (large)##################

# The fixtures bellow exist for the testing of HTTomo with raw projection data at Diamond.
# Jenkins CI at Diamond will provide an access to this test data. Otherwise
# this test data is not available to the user and the relevant tests that are marked as `full_data`
# will be ignored.


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
def i13_177906():
    # 2.5k projections, 27gb dataset
    return "tests/test_data/raw_data/i13/177906.nxs"


@pytest.fixture
def i13_179623():
    # 6k projections, 65gb dataset, 360 degrees scan
    return "tests/test_data/raw_data/i13/360/179623.nxs"


@pytest.fixture
def i12_119647():
    # 1.8k projections, 20gb dataset, 180 degrees scan
    return "tests/test_data/raw_data/i12/119647.nxs"


############## --Ground Truth references-- #################


@pytest.fixture
def FBP3d_tomobar_k11_38731_npz():
    # 10 slices numpy array
    return np.load("tests/test_data/raw_data/diad/FBP3d_tomobar_k11-38731.npz")


@pytest.fixture
def FBP3d_tomobar_k11_38730_npz():
    # 10 slices numpy array
    return np.load("tests/test_data/raw_data/diad/FBP3d_tomobar_k11-38730.npz")


@pytest.fixture
def FBP3d_tomobar_i12_119647_npz():
    # 10 slices numpy array
    return np.load("tests/test_data/raw_data/i12/FBP3d_tomobar_i12_119647.npz")


@pytest.fixture
def FBP2d_astra_i12_119647_npz():
    # 10 slices numpy array
    return np.load("tests/test_data/raw_data/i12/FBP2d_astra_i12_119647.npz")


@pytest.fixture
def FBP3d_tomobar_TVdenoising_i13_177906_npz():
    # 10 slices numpy array
    return np.load(
        "tests/test_data/raw_data/i13/FBP3d_tomobar_TVdenoising_i13_177906.npz"
    )


@pytest.fixture
def FBP3d_tomobar_paganin_i13_179623_npz():
    # 10 slices numpy array
    return np.load(
        "tests/test_data/raw_data/i13/360/FBP3d_tomobar_paganin_i13_179623.npz"
    )


@pytest.fixture
def FBP3d_tomobar_distortion_i13_179623_npz():
    # 10 slices numpy array
    return np.load(
        "tests/test_data/raw_data/i13/360/FBP3d_tomobar_distortion_i13_179623.npz"
    )


@pytest.fixture
def LPRec3d_tomobar_i12_119647_npz():
    # 10 slices numpy array
    return np.load("tests/test_data/raw_data/i12/LPRec3d_tomobar_i12_119647.npz")


@pytest.fixture
def pipeline_sweep_FBP3d_tomobar_i13_177906_tiffs():
    # 8 tiff files of 16bit
    return "tests/test_data/raw_data/i13/sweep/images_sweep_FBP3d_tomobar16bit_tif/"


@pytest.fixture
def pipeline_paganin_sweep_paganin_images_i12_119647_tiffs():
    # 3 tiff files from Paganin filter
    return (
        "tests/test_data/raw_data/i12/sweep/images_sweep_paganin_filter_tomopy8bit_tif/"
    )


@pytest.fixture
def pipeline_paganin_sweep_recon_images_i12_119647_tiffs():
    # 3 tiff files from reconstruction with paganin filter pipeline
    return "tests/test_data/raw_data/i12/sweep/images_sweep_FBP3d_tomobar16bit_tif/"


@pytest.fixture
def pipeline_parallel_titaren_center_pc_FBP3d_resample_i12_119647_tiffs():
    # 200 downsampled to 512 x 512 tiff files from reconstruction of i12_119647
    return "tests/test_data/raw_data/i12/tiffs/images8bit_tif/"


# ---------------------END------------------------#

# TODO: deprecate when loader is generalised


@pytest.fixture
def i12_data():
    return "tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"


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
        ignore=False,
    )


@pytest.fixture
def standard_data_ignore_darks_flats_config() -> DarksFlatsFileConfig:
    return DarksFlatsFileConfig(
        file=Path(__file__).parent / "test_data/tomo_standard.nxs",
        data_path="/entry1/tomo_entry/data/data",
        image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
        ignore=True,
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


def check_tif(files: List, number: int, shape: Tuple):
    # check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == number

    # check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    assert imarray.shape == shape


def compare_tif(files_list_to_compare: list, file_path_to_references: list):
    tif_files = sorted(list(filter(lambda x: ".tif" in x, files_list_to_compare)))
    tif_files_references = sorted(
        list(filter(lambda x: ".tif" in x, file_path_to_references))
    )

    for index in range(len(tif_files)):
        res_images = np.array(Image.open(tif_files[index])) - np.array(
            Image.open(tif_files_references[index])
        )
        res_norm = np.linalg.norm(res_images.flatten())
        assert res_norm < 1e-6


def change_value_parameters_method_pipeline(
    yaml_path: str,
    method: list,
    key: list,
    value: list,
    save_result: Union[None, bool] = None,
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
                if save_result is not None:
                    # add save_result to the list of keys
                    opened_yaml[i]["save_result"] = save_result

    with open(yaml_path, "w") as file_descriptor:
        yaml.dump(
            opened_yaml, file_descriptor, default_flow_style=False, sort_keys=False
        )
