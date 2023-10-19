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


@pytest.fixture
def dummy_dataset() -> DataSet:
    return DataSet(
        data=np.ones((10, 10, 10)),
        angles=np.ones((20,)),
        flats=3 * np.ones((10, 10)),
        darks=2 * np.ones((10, 10)),
    )


def test_basewrapper_execute_transfers_to_gpu(
    dummy_dataset: DataSet, mocker: MockerFixture
):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper("module_path", "fake_method", MPI.COMM_WORLD, True)
    dataset = wrp.execute(dummy_dataset)

    assert dataset.is_gpu == gpu_enabled


def test_basewrapper_execute_calls_pre_post_process(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper("module_path", "fake_method", MPI.COMM_WORLD, True)
    prep = mocker.patch.object(wrp, "_preprocess_data", return_value=dummy_dataset)
    post = mocker.patch.object(wrp, "_postprocess_data", return_value=dummy_dataset)
    trans = mocker.patch.object(wrp, "_transfer_data", return_value=dummy_dataset)

    wrp.execute(dummy_dataset)

    prep.assert_called_once_with(dummy_dataset)
    post.assert_called_once_with(dummy_dataset)
    trans.assert_called_once_with(dummy_dataset)


def test_wrapper_fails_with_wrong_returntype(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data):
            # returning None should not be allowed for generic wrapper,
            # it must be a cpu or gpu array (data)
            return None

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "fake_method", MPI.COMM_WORLD, False
    )
    with pytest.raises(ValueError) as e:
        wrp.execute(dummy_dataset)
    assert "return type" in str(e)


def test_wrapper_different_data_parameter_name(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(array):
            np.testing.assert_array_equal(array, 42)
            return array

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "fake_method", MPI.COMM_WORLD, False
    )
    dummy_dataset.data[:] = 42
    wrp.execute(dummy_dataset)


def test_wrapper_sets_config_params_constructor(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "param_tester", MPI.COMM_WORLD, False, param=42
    )
    wrp.execute(dummy_dataset)

    assert wrp["param"] == 42


def test_wrapper_sets_config_params_setter(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "param_tester", MPI.COMM_WORLD, False
    )
    wrp["param"] = 42
    wrp.execute(dummy_dataset)

    assert wrp["param"] == 42


def test_wrapper_sets_config_params_append_dict(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param1, param2):
            assert param1 == 42
            assert param2 == 43
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path", "param_tester", MPI.COMM_WORLD, False
    )
    wrp.append_config_params({"param1": 42, "param2": 43})
    wrp.execute(dummy_dataset)

    assert wrp["param1"] == 42
    assert wrp["param2"] == 43


def test_wrapper_passes_darks_flats_to_normalize(
    mocker: MockerFixture, dummy_dataset: DataSet
):
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
    dummy_dataset.unlock()
    dummy_dataset.data[:] = 1
    dummy_dataset.darks[:] = 2
    dummy_dataset.flats[:] = 3
    dummy_dataset.lock()

    wrp.execute(dummy_dataset)

    importmock.assert_called_once_with("mocked_module_path")


def test_wrapper_handles_reconstruction_angle_reshape(
    mocker: MockerFixture, dummy_dataset: DataSet
):
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
    dummy_dataset.data[:] = 1
    dummy_dataset.unlock()
    dummy_dataset.angles[:] = 2
    dummy_dataset.lock()

    wrp.execute(dummy_dataset)


def test_wrapper_rotation_180(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def rotation_tester(data):
            return 42.0  # center of rotation

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.rotation", "rotation_tester", MPI.COMM_WORLD, False
    )

    new_dataset = wrp.execute(dummy_dataset)

    assert wrp.get_side_output() == {"cor": 42.0}
    assert new_dataset == dummy_dataset  # note: not a deep comparison


def test_wrapper_rotation_360(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def rotation_tester(data):
            # cor, overlap, side, overlap_position - from find_center_360
            return 42.0, 3.0, 1, 10.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.rotation", "rotation_tester", MPI.COMM_WORLD, False
    )
    new_dataset = wrp.execute(dummy_dataset)

    assert wrp.get_side_output() == {
        "cor": 42.0,
        "overlap": 3.0,
        "side": 1,
        "overlap_position": 10.0,
    }
    assert new_dataset == dummy_dataset  # note: not a deep comparison


def test_wrapper_dezinging(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def remove_outlier3d(x):
            return 2 * x

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.prep", "remove_outlier3d", MPI.COMM_WORLD, False
    )
    dummy_dataset.unlock()
    dummy_dataset.data[:] = 1
    dummy_dataset.flats[:] = 3
    dummy_dataset.darks[:] = 4
    dummy_dataset.lock()

    newset = wrp.execute(dummy_dataset)

    # we double them all, so we expect all 3 to be twice the input
    np.testing.assert_array_equal(newset.data, 2)
    np.testing.assert_array_equal(newset.flats, 6)
    np.testing.assert_array_equal(newset.darks, 8)


def test_wrapper_save_to_images(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def save_to_images(data, out_dir, comm_rank, axis, file_format):
            np.testing.assert_array_equal(data, 1)
            assert out_dir == httomo.globals.run_out_dir
            assert comm_rank == MPI.COMM_WORLD.rank
            assert axis == 1
            assert file_format == "tif"

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.images",
        "save_to_images",
        MPI.COMM_WORLD,
        False,
        axis=1,
        file_format="tif",
    )
    newset = wrp.execute(dummy_dataset)

    assert newset == dummy_dataset


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.cupy
def test_wrapper_images_leaves_gpudata(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def save_to_images(data):
            assert getattr(data, "device", None) is None  # make sure it's on CPU

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        "mocked_module_path.images", "save_to_images", MPI.COMM_WORLD, False
    )
    with xp.cuda.Device(0):
        dummy_dataset.to_gpu()
        new_dataset = wrp.execute(dummy_dataset)

        assert new_dataset.is_gpu is True
