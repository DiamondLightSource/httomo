from typing import Union
from unittest.mock import MagicMock
import numpy as np
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.rotation import RotationWrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.utils import Pattern, gpu_enabled, xp
from ..testing_utils import make_mock_repo


import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_rotation_fails_with_sinogram_method(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data, ind=None):
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    with pytest.raises(NotImplementedError):
        make_method_wrapper(
            make_mock_repo(mocker, pattern=Pattern.sinogram),
            "mocked_module_path.rotation",
            "rotation_tester",
            MPI.COMM_WORLD,
        )


def test_rotation_accumulates_blocks(mocker: MockerFixture):
    GLOBAL_SHAPE = (10, 10, 30)
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )
    aux_data = AuxiliaryData(
        angles=np.ones(GLOBAL_SHAPE[0], dtype=np.float32),
        darks=2.0 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), np.float32),
        flats=3.0 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), np.float32),
    )
    b1 = DataSetBlock(
        data=global_data[0 : GLOBAL_SHAPE[0] // 2, :, :],
        aux_data=aux_data,
        block_start=0,
        global_shape=GLOBAL_SHAPE,
        chunk_shape=GLOBAL_SHAPE,
    )
    b2 = DataSetBlock(
        data=global_data[GLOBAL_SHAPE[0] // 2 :, :, :],
        aux_data=aux_data,
        block_start=GLOBAL_SHAPE[0] // 2,
        global_shape=GLOBAL_SHAPE,
        chunk_shape=GLOBAL_SHAPE,
    )

    class FakeModule:
        def rotation_tester(data, ind=None):
            assert data.ndim == 2  # for 1 slice only
            np.testing.assert_array_equal(
                data, global_data[:, (GLOBAL_SHAPE[1] - 1) // 2, :]
            )
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        output_mapping={"cor": "cor"},
    )
    assert isinstance(wrp, RotationWrapper)
    normalize = mocker.patch.object(
        wrp, "normalize_sino", side_effect=lambda sino, flats, darks: sino
    )

    wrp.execute(b1)
    normalize.assert_not_called()
    wrp.execute(b2)
    normalize.assert_called_once()
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
        def rotation_tester(data, ind=None):
            assert rank == 0  # for rank 1, it shouldn't be called
            assert data.ndim == 2  # for 1 slice only
            assert ind == 0
            if ind_par == "mid" or ind_par is None:
                xp.testing.assert_array_equal(
                    global_data[:, (GLOBAL_SHAPE[1] - 1) // 2, :],
                    data,
                )
            else:
                xp.testing.assert_array_equal(global_data[:, ind_par, :], data)
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    comm = mocker.MagicMock()
    comm.rank = rank
    comm.size = 2
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection, implementation="gpu_cupy"),
        "mocked_module_path.rotation",
        "rotation_tester",
        comm,
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
    normalize = mocker.patch.object(
        wrp, "normalize_sino", side_effect=lambda sino, f, d: sino
    )

    comm.bcast.return_value = 42.0

    res = wrp.execute(block)

    assert wrp.pattern == Pattern.projection
    xp.testing.assert_array_equal(res.data, rank0_data if rank == 0 else rank1_data)
    comm.bcast.assert_called_once()
    if rank == 0:
        normalize.assert_called_once()
    else:
        normalize.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_rotation_gather_sino_slice(mocker: MockerFixture, rank: int):
    class FakeModule:
        def rotation_tester(data, ind=None):
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    comm = mocker.MagicMock()
    comm.rank = rank
    comm.size = 2
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        comm,
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


def test_rotation_normalize_sino_no_darks_flats():
    self_mock = MagicMock()  # this mocks self - as function only sets gpu time
    ret = RotationWrapper.normalize_sino(
        self_mock, np.ones((10, 10), dtype=np.float32), None, None
    )

    assert ret.shape == (10, 1, 10)
    np.testing.assert_allclose(np.squeeze(ret), 1.0)


def test_rotation_normalize_sino_same_darks_flats():
    self_mock = MagicMock()  # this mocks self - as function only sets gpu time
    ret = RotationWrapper.normalize_sino(
        self_mock,
        np.ones((10, 10), dtype=np.float32),
        0.5
        * np.ones(
            (
                10,
                10,
            ),
            dtype=np.float32,
        ),
        0.5
        * np.ones(
            (
                10,
                10,
            ),
            dtype=np.float32,
        ),
    )

    assert ret.shape == (10, 1, 10)
    np.testing.assert_allclose(ret, 0.5)


def test_rotation_normalize_sino_scalar():
    self_mock = MagicMock()  # this mocks self - as function only sets gpu time
    ret = RotationWrapper.normalize_sino(
        self_mock,
        np.ones((10, 10), dtype=np.float32),
        0.5
        * np.ones(
            (
                10,
                10,
            ),
            dtype=np.float32,
        ),
        0.5
        * np.ones(
            (
                10,
                10,
            ),
            dtype=np.float32,
        ),
    )

    assert ret.shape == (10, 1, 10)
    np.testing.assert_allclose(ret, 0.5)


def test_rotation_normalize_sino_different_darks_flats():
    self_mock = MagicMock()  # this mocks self - as function only sets gpu time
    ret = RotationWrapper.normalize_sino(
        self_mock,
        2.0 * np.ones((10, 10), dtype=np.float32),
        1.0 * np.ones((10, 10), dtype=np.float32),
        0.5 * np.ones((10, 10), dtype=np.float32),
    )

    assert ret.shape == (10, 1, 10)
    np.testing.assert_allclose(np.squeeze(ret), 1.0)


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.cupy
def test_rotation_normalize_sino_different_darks_flats_gpu():
    self_mock = MagicMock()  # this mocks self - as function only sets gpu time
    ret = RotationWrapper.normalize_sino(
        self_mock,
        2.0 * xp.ones((10, 10), dtype=np.float32),
        1.0 * xp.ones((10, 10), dtype=np.float32),
        0.5 * xp.ones((10, 10), dtype=np.float32),
    )

    assert ret.shape == (10, 1, 10)
    assert getattr(ret, "device", None) is not None
    xp.testing.assert_allclose(xp.squeeze(ret), 1.0)


def test_rotation_180(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data, ind):
            return 42.0  # center of rotation

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        output_mapping={"cor": "center"},
        ind=5,
    )

    block = DataSetBlock(
        data=np.ones((10, 10, 10), dtype=np.float32),
        aux_data=AuxiliaryData(angles=np.ones(10, dtype=np.float32)),
    )
    new_block = wrp.execute(block)

    assert wrp.get_side_output() == {"center": 42.0}
    assert new_block == block  # note: not a deep comparison


def test_rotation_pc_180(mocker: MockerFixture):
    class FakeModule:
        def find_center_pc(proj1, proj2=None):
            return 42.0  # center of rotation

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "find_center_pc",
        MPI.COMM_WORLD,
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
        def rotation_tester(data, ind):
            # cor, overlap, side, overlap_position - from find_center_360
            return 42.0, 3.0, 1, 10.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.projection),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        output_mapping={
            "cor": "center",
            "overlap": "overlap",
            "overlap_position": "pos",
        },
        ind=5,
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
