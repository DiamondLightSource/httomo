from typing import Any, Dict, List, Tuple, Union
from unittest.mock import MagicMock
import numpy as np
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.rotation import RotationWrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.utils import gpu_enabled, make_3d_shape_from_shape, xp
from ..testing_utils import make_mock_preview_config, make_mock_repo

from httomo_backends.methods_database.query import Pattern

import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_rotation_fails_with_sinogram_method(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data, ind=None):
            return 42.0

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    with pytest.raises(NotImplementedError):
        make_method_wrapper(
            make_mock_repo(mocker, pattern=Pattern.sinogram),
            "mocked_module_path.rotation",
            "rotation_tester",
            MPI.COMM_WORLD,
            make_mock_preview_config(mocker),
        )


@pytest.mark.parametrize(
    "padding",
    [(0, 0), (2, 3)],
    ids=["zero-padding", "non-zero-padding"],
)
def test_rotation_accumulates_blocks(mocker: MockerFixture, padding: Tuple[int, int]):
    GLOBAL_SHAPE = (10, 10, 30)
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )
    aux_data = AuxiliaryData(
        angles=np.ones(GLOBAL_SHAPE[0], dtype=np.float32),
        darks=2.0 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), np.float32),
        flats=3.0 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), np.float32),
    )

    chunk_shape_list: List[int] = list(GLOBAL_SHAPE)
    if padding != (0, 0):
        chunk_shape_list[0] += padding[0] + padding[1]

    b1_data = global_data[0 : GLOBAL_SHAPE[0] // 2, :, :]
    b1_block_start = 0
    if padding != (0, 0):
        b1_data = np.pad(b1_data, pad_width=(padding, (0, 0), (0, 0)))
        b1_block_start -= padding[0]
    b1 = DataSetBlock(
        data=b1_data,
        aux_data=aux_data,
        block_start=b1_block_start,
        chunk_start=0 if padding == (0, 0) else -padding[0],
        global_shape=GLOBAL_SHAPE,
        chunk_shape=make_3d_shape_from_shape(chunk_shape_list),
        padding=padding,
    )

    b2_data = global_data[GLOBAL_SHAPE[0] // 2 :, :, :]
    b2_block_start = GLOBAL_SHAPE[0] // 2
    if padding != (0, 0):
        b2_data = np.pad(b2_data, pad_width=(padding, (0, 0), (0, 0)))
        b2_block_start -= padding[0]
    b2 = DataSetBlock(
        data=b2_data,
        aux_data=aux_data,
        block_start=b2_block_start,
        chunk_start=0 if padding == (0, 0) else -padding[0],
        global_shape=GLOBAL_SHAPE,
        chunk_shape=make_3d_shape_from_shape(chunk_shape_list),
        padding=padding,
    )

    class FakeModule:
        def rotation_tester(data, ind=None, average_radius=None):
            assert data.ndim == 3  # for 1 slice only
            np.testing.assert_array_equal(
                data[:, 0, :], global_data[:, (GLOBAL_SHAPE[1] - 1) // 2, :]
            )
            return 42.0

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={"cor": "cor"},
    )
    assert isinstance(wrp, RotationWrapper)
    wrp.execute(b1)
    wrp.execute(b2)
    assert wrp.get_side_output() == {"cor": 42.0}


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("ind_par", ["mid", 2, None])
@pytest.mark.cupy
def test_rotation_gathers_single_sino_slice(
    mocker: MockerFixture,
    rank: int,
    ind_par: Union[str, int, None],
    gpu: bool,
):
    GLOBAL_SHAPE = (10, 10, 30)
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )
    rank0_data = global_data[: GLOBAL_SHAPE[0] // 2, :, :]
    rank1_data = global_data[GLOBAL_SHAPE[0] // 2 :, :, :]
    aux_data = AuxiliaryData(
        angles=np.ones(GLOBAL_SHAPE[0], dtype=np.float32),
        darks=2.0 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), np.float32),
        flats=3.0 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), np.float32),
    )
    block = DataSetBlock(
        data=rank0_data if rank == 0 else rank1_data,
        aux_data=aux_data,
        slicing_dim=0,
        block_start=0,
        chunk_start=0 if rank == 0 else GLOBAL_SHAPE[1] // 2,
        global_shape=GLOBAL_SHAPE,
        chunk_shape=(GLOBAL_SHAPE[0] // 2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]),
    )

    class FakeModule:
        def rotation_tester(data, ind=None, average_radius=None):
            assert rank == 0  # for rank 1, it shouldn't be called
            assert data.ndim == 3  # for 1 slice only
            assert ind == 0
            assert average_radius == 0
            if ind_par == "mid" or ind_par is None:
                xp.testing.assert_array_equal(
                    global_data[:, (GLOBAL_SHAPE[1] - 1) // 2, :],
                    data[:, ind, :],
                )
            else:
                xp.testing.assert_array_equal(global_data[:, ind_par, :], data[:, 0, :])
            return 42.0

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    comm = mocker.MagicMock()
    comm.rank = rank
    comm.size = 2
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection, implementation="gpu_cupy"),
        "mocked_module_path.rotation",
        "rotation_tester",
        comm,
        make_mock_preview_config(mocker),
    )
    assert isinstance(wrp, RotationWrapper)

    if ind_par is not None:
        wrp["ind"] = ind_par

    if gpu:
        block.to_gpu()

    if ind_par == "mid" or ind_par is None:
        sino_slice = global_data[:, (GLOBAL_SHAPE[1] - 1) // 2, :]
    else:
        assert isinstance(ind_par, int)
        sino_slice = global_data[:, ind_par, :]
    mocker.patch.object(wrp, "_gather_sino_slice", side_effect=lambda s: sino_slice)

    comm.bcast.return_value = 42.0

    res = wrp.execute(block)

    assert wrp.pattern == Pattern.projection
    xp.testing.assert_array_equal(res.data, rank0_data if rank == 0 else rank1_data)
    comm.bcast.assert_called_once()


@pytest.mark.parametrize("rank", [0, 1])
def test_rotation_gather_sino_slice(mocker: MockerFixture, rank: int):
    class FakeModule:
        def rotation_tester(data, ind=None):
            return 42.0

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    comm = mocker.MagicMock()
    comm.rank = rank
    comm.size = 2
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        comm,
        make_mock_preview_config(mocker),
    )
    assert isinstance(wrp, RotationWrapper)
    if rank == 0:
        wrp.sino = np.arange(2 * 6, dtype=np.float32).reshape((2, 6))
    else:
        wrp.sino = np.arange(2 * 6, 5 * 6, dtype=np.float32).reshape((3, 6))

    if rank == 0:
        comm.gather.return_value = [2 * 6, 3 * 6]
    else:
        comm.gather.return_value = [2 * 6]

    res = wrp._gather_sino_slice((5, 13, 6))

    comm.Gatherv.assert_called_once()
    if rank == 0:
        assert res.shape == (5, 6)
        comm.gather.assert_called_once_with(2 * 6)
    else:
        assert res is None
        comm.gather.assert_called_once_with(3 * 6)


def test_rotation_180(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data, ind, average_radius):
            return 42.0  # center of rotation

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={"cor": "center"},
        ind=5,
        average_radius=0,
    )

    block = DataSetBlock(
        data=np.ones((10, 10, 10), dtype=np.float32),
        aux_data=AuxiliaryData(angles=np.ones(10, dtype=np.float32)),
    )
    new_block = wrp.execute(block)

    assert wrp.get_side_output() == {"center": 42.0}
    assert new_block == block  # note: not a deep comparison


@pytest.mark.parametrize(
    "radius",
    [5, 6],
)
def test_rotation_180_raise_average_radius(mocker: MockerFixture, radius):
    class FakeModule:
        def rotation_tester(data, ind, average_radius):
            return 42.0  # center of rotation

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={"cor": "center"},
        ind=5,
        average_radius=radius,
    )

    block = DataSetBlock(
        data=np.ones((10, 10, 10), dtype=np.float32),
        aux_data=AuxiliaryData(angles=np.ones(10, dtype=np.float32)),
    )
    if radius == 5:
        with pytest.raises(ValueError) as e:
            _ = wrp.execute(block)
        assert (
            "The given average_radius = 5 in the centering method is larger or equal than the half size of the block = 5."
            in str(e)
        )
    if radius == 6:
        with pytest.raises(ValueError) as e:
            _ = wrp.execute(block)
        assert (
            "The given average_radius = 6 in the centering method is larger or equal than the half size of the block = 5."
            in str(e)
        )


def test_rotation_180_average_within_range(mocker: MockerFixture):
    IND = 3
    AVERAGE_RADIUS = 1
    original_array = np.zeros((7, 7, 7), dtype=np.float32)
    original_array[:, 2, :] = 1.2
    original_array[:, 3, :] = 2.5
    original_array[:, 4, :] = 13.7
    expected_data = np.mean(
        original_array[:, IND - AVERAGE_RADIUS : IND + AVERAGE_RADIUS + 1, :],
        axis=1,
    )

    class FakeModule:
        def rotation_tester(data, ind, average_radius):
            np.testing.assert_array_equal(data, expected_data[:, np.newaxis, :])
            return 100

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={"cor": "center"},
        ind=IND,
        average_radius=AVERAGE_RADIUS,
    )

    block = DataSetBlock(
        data=original_array,
        aux_data=AuxiliaryData(angles=np.ones(7, dtype=np.float32)),
    )
    wrp.execute(block)


@pytest.mark.parametrize(
    "params",
    [{"ind": 3, "average_radius": 0}, {"ind": 3}],
    ids=[
        "explicitly-sets-0",
        "wrapper-implicitly-sets-0",
    ],
)
def test_rotation_180_average_0slices(mocker: MockerFixture, params: Dict[str, Any]):
    original_array = np.zeros((7, 7, 7), dtype=np.float32)
    original_array[:, 2, :] = 1.2
    original_array[:, 3, :] = 2.5
    original_array[:, 4, :] = 13.7
    expected_data = original_array[:, params["ind"], :]

    class FakeModule:
        def rotation_tester(data, ind, average_radius):
            np.testing.assert_array_equal(data, expected_data[:, np.newaxis, :])
            return 100

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={"cor": "center"},
        **params,
    )

    block = DataSetBlock(
        data=original_array,
        aux_data=AuxiliaryData(angles=np.ones(7, dtype=np.float32)),
    )
    wrp.execute(block)


def test_rotation_pc_180(mocker: MockerFixture):
    class FakeModule:
        def find_center_pc(proj1, proj2=None):
            return 42.0  # center of rotation

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "find_center_pc",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={"cor": "center"},
    )

    block = DataSetBlock(
        data=np.ones((10, 10, 10), dtype=np.float32),
        aux_data=AuxiliaryData(angles=np.ones(10, dtype=np.float32)),
    )
    new_block = wrp.execute(block)

    assert wrp.get_side_output() == {"center": 42.0}
    assert new_block == block  # note: not a deep comparison


def test_rotation_360(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data, ind, average_radius):
            # cor, overlap, side, overlap_position - from find_center_360
            return 42.0, 3.0, 1, 10.0

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={
            "cor": "center",
            "overlap": "overlap",
            "overlap_position": "pos",
        },
        ind=5,
        average_radius=0,
    )
    block = DataSetBlock(
        data=np.ones((10, 10, 10), dtype=np.float32),
        aux_data=AuxiliaryData(angles=np.ones(10, dtype=np.float32)),
    )
    new_block = wrp.execute(block)

    assert wrp.get_side_output() == {
        "center": 42.0,
        "overlap": 3.0,
        "pos": 10.0,
    }
    assert new_block == block  # note: not a deep comparison
