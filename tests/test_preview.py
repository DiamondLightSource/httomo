from pathlib import Path
from typing import Tuple
from unittest import mock
import pytest
import h5py
import numpy as np
from pytest_mock import MockerFixture

from httomo.preview import Preview, PreviewConfig, PreviewDimConfig
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.runner.loader import LoaderInterface
from httomo_backends.methods_database.query import Pattern
from typing import Literal, Optional
from httomo.method_wrappers import make_method_wrapper
from tests.testing_utils import (
    make_mock_preview_config,
    make_mock_repo,
)
from mpi4py import MPI
from httomo_backends.methods_database.query import GpuMemoryRequirement

from httomo.ui_layer import fix_preview_y_if_smaller_than_padding


@pytest.mark.parametrize(
    "preview_config, is_error_expected, err_str",
    [
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=221),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            "Preview indices in angles dim exceed bounds of data: start=0, stop=221",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=129),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            "Preview indices in detector_y dim exceed bounds of data: start=0, stop=129",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=161),
            ),
            True,
            "Preview indices in detector_x dim exceed bounds of data: start=0, stop=161",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=220, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            (
                "Preview index error for angles: start must be strictly smaller than "
                "stop, but start=220, stop=220"
            ),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=60, stop=50),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            (
                "Preview index error for detector_y: start must be strictly smaller than "
                "stop, but start=60, stop=50"
            ),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=50, stop=0),
            ),
            True,
            (
                "Preview index error for detector_x: start must be strictly smaller than "
                "stop, but start=50, stop=0"
            ),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            False,
            "",
        ),
    ],
    ids=[
        "incorrect_angles_bounds",
        "incorrect_det_y_bounds",
        "incorrect_det_x_bounds",
        "start_geq_stop_det_angles_bounds",
        "start_geq_stop_det_y_bounds",
        "start_geq_stop_det_x_bounds",
        "all_correct_bounds",
    ],
)
def test_preview_bound_checking(
    standard_data_path: str,
    standard_image_key_path: str,
    preview_config: PreviewConfig,
    is_error_expected: bool,
    err_str: str,
):
    IN_FILE = Path(__file__).parent / "test_data/tomo_standard.nxs"
    f = h5py.File(IN_FILE, "r")
    dataset = f[standard_data_path]
    image_key = f[standard_image_key_path]

    if is_error_expected:
        with pytest.raises(ValueError, match=err_str):
            _ = Preview(
                preview_config=preview_config,
                dataset=dataset,
                image_key=image_key,
            )
    else:
        preview = Preview(
            preview_config=preview_config,
            dataset=dataset,
            image_key=image_key,
        )
        assert preview.config == preview_config

    f.close()


def test_preview_calculate_data_indices_excludes_darks_flats(
    standard_data_path: str,
    standard_image_key_path: str,
):
    IN_FILE_PATH = Path(__file__).parent / "test_data/tomo_standard.nxs"
    f = h5py.File(IN_FILE_PATH, "r")
    dataset = f[standard_data_path]
    image_key = f[standard_image_key_path]
    all_indices: np.ndarray = image_key[:]
    data_indices = np.where(all_indices == 0)[0]

    config = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=220),
        detector_y=PreviewDimConfig(start=0, stop=128),
        detector_x=PreviewDimConfig(start=0, stop=160),
    )
    preview = Preview(
        preview_config=config,
        dataset=dataset,
        image_key=image_key,
    )
    assert not np.array_equal(preview.data_indices, all_indices)
    assert np.array_equal(preview.data_indices, data_indices)
    assert preview.config.angles == PreviewDimConfig(start=0, stop=180)
    f.close()


def test_preview_with_no_image_key():
    IN_FILE_PATH = (
        Path(__file__).parent
        / "test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
    )
    f = h5py.File(IN_FILE_PATH, "r")
    DATA_PATH = "1-TempPlugin-tomo/data"
    ANGLES, DET_Y, DET_X = (724, 10, 192)
    expected_indices = list(range(ANGLES))
    config = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=ANGLES),
        detector_y=PreviewDimConfig(start=0, stop=DET_Y),
        detector_x=PreviewDimConfig(start=0, stop=DET_X),
    )
    preview = Preview(
        preview_config=config,
        dataset=f[DATA_PATH],
        image_key=None,
    )
    assert np.array_equal(preview.data_indices, expected_indices)


@pytest.mark.parametrize(
    "previewed_shape",
    [(100, 128, 160), (180, 10, 160), (180, 128, 10)],
    ids=["crop_angles_dim", "crop_det_y_dim", "crop_det_x_dim"],
)
def test_preview_global_shape(
    standard_data_path: str,
    standard_image_key_path: str,
    previewed_shape: Tuple[int, int, int],
):
    IN_FILE_PATH = Path(__file__).parent / "test_data/tomo_standard.nxs"
    f = h5py.File(IN_FILE_PATH, "r")
    dataset = f[standard_data_path]
    image_key = f[standard_image_key_path]

    config = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=previewed_shape[0]),
        detector_y=PreviewDimConfig(start=0, stop=previewed_shape[1]),
        detector_x=PreviewDimConfig(start=0, stop=previewed_shape[2]),
    )
    preview = Preview(
        preview_config=config,
        dataset=dataset,
        image_key=image_key,
    )
    assert preview.global_shape == previewed_shape


@pytest.mark.parametrize(
    "detY_preview_stop",
    [18, 19, 20, 21],
)
def tests_preview_modifier_padding(mocker: MockerFixture, detY_preview_stop: int):
    detY_preview_start = 10
    detY_preview_stop = detY_preview_stop
    slices_total = detY_preview_stop - detY_preview_start
    EXPECTED_PADDING = (5, 5)
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (1800, slices_total, 160)
    data = np.arange(np.prod(PREVIEWED_SLICES_SHAPE), dtype=np.uint16).reshape(
        PREVIEWED_SLICES_SHAPE
    )
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    preview = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[0]),
        detector_y=PreviewDimConfig(start=detY_preview_start, stop=detY_preview_stop),
        detector_x=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[2]),
    )

    class FakeModule:
        def total_variation_PD(data: np.ndarray, regularisation_parameter: float, iterations: int):  # type: ignore
            return data

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface,
        instance=True,
        pattern=Pattern.all,
        method_name="testloader",
        reslice=False,
        preview=preview,
    )

    def mock_make_data_source(padding) -> DataSetSource:
        ret = mocker.create_autospec(
            DataSetSource,
            global_shape=block.global_shape,
            dtype=block.data.dtype,
            chunk_shape=block.chunk_shape,
            chunk_index=block.chunk_index,
            slicing_dim=1 if loader.pattern == Pattern.sinogram else 0,
            aux_data=block.aux_data,
            preview=preview,
        )
        type(ret).raw_shape = mock.PropertyMock(return_value=GLOBAL_SHAPE)
        slicing_dim: Literal[0, 1, 2] = 0
        mocker.patch.object(
            ret,
            "read_block",
            side_effect=lambda start, length: DataSetBlock(
                data=block.data[start : start + length, :, :],
                aux_data=block.aux_data,
                global_shape=block.global_shape,
                chunk_shape=block.chunk_shape,
                slicing_dim=slicing_dim,
                block_start=start,
                chunk_start=block.chunk_index[slicing_dim],
            ),
        )
        return ret

    mocker.patch.object(
        loader,
        "make_data_source",
        side_effect=mock_make_data_source,
    )

    total_variation_PD_params = {
        "regularisation_parameter": 1.5e-4,
        "iterations": 100,
    }
    memory_gpu: Optional[GpuMemoryRequirement] = None
    repo = make_mock_repo(
        mocker,
        pattern=Pattern.all,
        implementation="gpu_cupy",
        memory_gpu=memory_gpu,
        padding=True,
    )
    padding_calc_mock = mocker.patch.object(
        repo.query("", ""), "calculate_padding", return_value=EXPECTED_PADDING
    )
    wrp = make_method_wrapper(
        repo,
        module_path="mocked_module_path.misc",
        method_name="total_variation_PD",
        comm=MPI.COMM_WORLD,
        preview_config=make_mock_preview_config(mocker),
        save_result=None,
        output_mapping={},
        **total_variation_PD_params,
    )
    loader = fix_preview_y_if_smaller_than_padding(loader, [wrp])
    padding = wrp.calculate_padding()

    if slices_total <= sum(padding):
        EXPECTED_PREVIEW_CONFIG = PreviewDimConfig(
            start=detY_preview_start - padding[0], stop=detY_preview_stop + padding[1]
        )
    else:
        EXPECTED_PREVIEW_CONFIG = PreviewDimConfig(
            start=detY_preview_start, stop=detY_preview_stop
        )

    assert loader.preview.detector_y == EXPECTED_PREVIEW_CONFIG
