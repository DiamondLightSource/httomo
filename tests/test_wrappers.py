import inspect
from mpi4py import MPI
import numpy as np
import httomo
from httomo.dataset import DataSet
from httomo.utils import xp, gpu_enabled
from pytest_mock import MockerFixture
import pytest

from httomo.wrappers_class import BackendWrapper, make_backend_wrapper


def test_tomopy_wrapper():
    wrp = BackendWrapper("tomopy", "recon", "algorithm", "recon", MPI.COMM_WORLD)
    assert inspect.ismodule(wrp.module)


def test_httomolib_wrapper():
    wrp = BackendWrapper(
        "httomolib", "misc", "images", "save_to_images", MPI.COMM_WORLD
    )
    assert inspect.ismodule(wrp.module)


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.cupy
def test_httomolibgpu_wrapper():
    wrp = BackendWrapper(
        "httomolibgpu", "prep", "normalize", "normalize", MPI.COMM_WORLD
    )
    assert inspect.ismodule(wrp.module)


# def test_httomolibgpu_wrapper_max_slices_gpu():
#     wrp = HttomolibgpuWrapper("prep", "normalize", "normalize", MPI.COMM_WORLD)
#     assert wrp.cupyrun is True
#     assert wrp.calc_max_slices(0, (100, 100), np.uint8(), 50000)[0] < 100000


# def test_httomolibgpu_wrapper_max_slices_passes_kwargs():
#     from httomolibgpu.prep.normalize import normalize

#     mock_method = mock.Mock()
#     mockMeta = dataclasses.replace(normalize.meta, calc_max_slices=mock_method)
#     with mock.patch.object(normalize, "meta", mockMeta):
#         wrp = HttomolibgpuWrapper("prep", "normalize", "normalize", MPI.COMM_WORLD)
#         wrp.dict_params = dict(testarg=1, minus_log=True)
#         wrp.calc_max_slices(0, (100, 100), np.uint8(), 50000)

#     # make sure the default args are called and the args given above are overriding the defaults
#     mock_method.assert_called_once_with(
#         0,
#         (100, 100),
#         np.uint8(),
#         50000,
#         cutoff=10.0,
#         minus_log=True,
#         testarg=1,
#         nonnegativity=False,
#         remove_nans=False,
#     )


# simple test method that can be passed to httomo
def method_tester(data: xp.ndarray) -> xp.ndarray:
    return data


def test_basewrapper_execute_transfers_to_gpu(mocker: MockerFixture):
    dataset = DataSet(
        np.ones((10, 10, 10)), np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))
    )
    wrp = make_backend_wrapper(
        "tests.test_wrappers", "method_tester", MPI.COMM_WORLD, True
    )
    dataset = wrp.execute(dict(), dataset, False)

    assert dataset.is_gpu == gpu_enabled


def test_basewrapper_execute_returns_to_cpu(mocker: MockerFixture):
    dataset = DataSet(
        np.ones((10, 10, 10)), np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))
    )
    wrp = make_backend_wrapper(
        "tests.test_wrappers", "method_tester", MPI.COMM_WORLD, True
    )
    dataset = wrp.execute(dict(), dataset, True)

    assert dataset.is_gpu is False


def test_basewrapper_execute_calls_pre_post_process(mocker: MockerFixture):
    dataset = DataSet(
        np.ones((10, 10, 10)), np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))
    )
    wrp = make_backend_wrapper(
        "tests.test_wrappers", "method_tester", MPI.COMM_WORLD, True
    )
    prep = mocker.patch.object(wrp, "_preprocess_data", return_value=dataset)
    post = mocker.patch.object(wrp, "_postprocess_data", return_value=dataset)
    trans = mocker.patch.object(wrp, "_transfer_data", return_value=dataset)

    wrp.execute(dict(), dataset, True)

    prep.assert_called_once_with(dataset)
    post.assert_called_once_with(dataset)
    trans.assert_called_once_with(dataset)


def test_wrapper_fails_with_wrong_returntype(mocker: MockerFixture):
    class FakeModule:
        def fake_method(data):
            # returning None should not be allowed for generic wrapper,
            # it must be a cpu or gpu array (data)
            return None

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "fake_method", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=np.ones((10, 10)),
        flats=3 * np.ones((10, 10)),
        darks=2 * np.ones((10, 10)),
    )
    with pytest.raises(ValueError) as e:
        wrp.execute(dict(), dataset, True)
    assert "return type" in str(e)


def test_wrapper_different_data_parameter_name(mocker: MockerFixture):
    class FakeModule:
        def fake_method(array):
            np.testing.assert_array_equal(array, 42)
            return array

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "fake_method", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=42 * np.ones((10, 10, 10)),
        angles=np.ones((10, 10)),
        flats=3 * np.ones((10, 10)),
        darks=2 * np.ones((10, 10)),
    )
    wrp.execute(dict(), dataset, True)


def test_wrapper_passes_darks_flats_to_normalize(mocker: MockerFixture):
    # we use getattr on the module to get the function,
    # so this fake works fine for tests:
    #   getattr(FakeModule, "normalize_tester") returns the right function
    class FakeModule:
        def normalize_tester(data, flats, darks):
            np.testing.assert_array_equal(data, 1)
            np.testing.assert_array_equal(darks, 2)
            np.testing.assert_array_equal(flats, 3)
            return data

    importmock = mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "normalize_tester", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=np.ones((10, 10)),
        flats=3 * np.ones((10, 10)),
        darks=2 * np.ones((10, 10)),
    )
    wrp.execute(dict(), dataset, True)

    importmock.assert_called_once_with("mocked_module_path")


def test_wrapper_handles_reconstruction_angle_reshape(mocker: MockerFixture):
    class FakeModule:
        def recon_tester(data, angles_radians):
            np.testing.assert_array_equal(data, 1)
            np.testing.assert_array_equal(angles_radians, 2)
            assert data.shape[0] == len(angles_radians)
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.algorithm", "recon_tester", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=2 * np.ones((20,)),
        flats=3 * np.ones((10, 10)),
        darks=4 * np.ones((10, 10)),
    )
    wrp.execute(dict(), dataset, True)


def test_wrapper_rotation_180(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data):
            return 42.0  # center of rotation

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.rotation", "rotation_tester", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=2 * np.ones((20,)),
        flats=3 * np.ones((10, 10)),
        darks=4 * np.ones((10, 10)),
    )
    new_dataset = wrp.execute(dict(), dataset, True)

    assert wrp.get_side_output() == {"cor": 42.0}
    assert new_dataset == dataset  # note: not a deep comparison


def test_wrapper_rotation_360(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data):
            # cor, overlap, side, overlap_position - from find_center_360
            return 42.0, 3.0, 1, 10.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.rotation", "rotation_tester", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=2 * np.ones((20,)),
        flats=3 * np.ones((10, 10)),
        darks=4 * np.ones((10, 10)),
    )
    new_dataset = wrp.execute(dict(), dataset, True)

    assert wrp.get_side_output() == {
        "cor": 42.0,
        "overlap": 3.0,
        "side": 1,
        "overlap_position": 10.0,
    }
    assert new_dataset == dataset  # note: not a deep comparison


def test_wrapper_dezinging(mocker: MockerFixture):
    class FakeModule:
        def remove_outlier3d(x):
            return 2 * x

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.prep", "remove_outlier3d", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=2 * np.ones((20,)),
        flats=3 * np.ones((10, 10)),
        darks=4 * np.ones((10, 10)),
    )
    newset = wrp.execute(dict(), dataset, True)

    # we double them all, so we expect all 3 to be twice the input
    np.testing.assert_array_equal(newset.data, 2)
    np.testing.assert_array_equal(newset.darks, 8)
    np.testing.assert_array_equal(newset.flats, 6)


def test_wrapper_save_to_images(mocker: MockerFixture):
    class FakeModule:
        def save_to_images(data, out_dir, comm_rank, axis, file_format):
            np.testing.assert_array_equal(data, 1)
            assert out_dir == httomo.globals.run_out_dir
            assert comm_rank == 0
            assert axis == 1
            assert file_format == "tif"

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.images", "save_to_images", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=2 * np.ones((20,)),
        flats=3 * np.ones((10, 10)),
        darks=4 * np.ones((10, 10)),
    )
    newset = wrp.execute(dict(axis=1, file_format="tif"), dataset, True)

    assert newset == dataset


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.cupy
@pytest.mark.parametrize("return_numpy", [False, True])
def test_wrapper_images_leaves_gpudata(mocker: MockerFixture, return_numpy: bool):
    class FakeModule:
        def save_to_images(data):
            assert getattr(data, "device", None) is None  # make sure it's on CPU

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.images", "save_to_images", MPI.COMM_WORLD, False
    )
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        angles=2 * np.ones((20,)),
        flats=3 * np.ones((10, 10)),
        darks=4 * np.ones((10, 10)),
    )
    with xp.cuda.Device(0):
        dataset.to_gpu()
        new_dataset = wrp.execute(dict(), dataset, return_numpy)

        assert new_dataset.is_gpu is not return_numpy
