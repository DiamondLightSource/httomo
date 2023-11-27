from typing import List, Optional, Union
from unittest.mock import ANY
from mpi4py import MPI
import numpy as np
import httomo
from httomo.runner.dataset import DataSet
from httomo.runner.backend_wrapper import RotationWrapper, make_backend_wrapper
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.output_ref import OutputRef
from httomo.utils import Pattern, xp, gpu_enabled
from pytest_mock import MockerFixture
import pytest

from .testing_utils import make_mock_repo, make_test_method


def test_basewrapper_get_name_and_paths(mocker: MockerFixture):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "testmodule.path",
        "fake_method",
        MPI.COMM_WORLD,
    )

    assert wrp.method_name == "fake_method"
    assert wrp.module_path == "testmodule.path"
    assert wrp.package_name == "testmodule"


def test_basewrapper_execute_transfers_to_gpu(
    dummy_dataset: DataSet, mocker: MockerFixture
):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "module_path",
        "fake_method",
        MPI.COMM_WORLD,
    )
    dataset = wrp.execute(dummy_dataset)

    assert dataset.is_gpu == gpu_enabled


def test_basewrapper_execute_calls_pre_post_process(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "module_path",
        "fake_method",
        MPI.COMM_WORLD,
    )
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
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    with pytest.raises(ValueError) as e:
        wrp.execute(dummy_dataset)
    assert "return type" in str(e)


def test_wrapper_sets_gpuid(mocker: MockerFixture, dummy_dataset: DataSet):
    mocker.patch("httomo.runner.backend_wrapper.gpu_enabled", True)

    class FakeModule:
        def fake_method(data, gpu_id: int):
            assert gpu_id == 4
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.gpu_id = 4
    wrp.execute(dummy_dataset)


def test_wrapper_fails_for_gpumethods_with_no_gpu(mocker: MockerFixture):
    mocker.patch("httomo.runner.backend_wrapper.gpu_enabled", False)

    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    with pytest.raises(ValueError) as e:
        make_backend_wrapper(
            make_mock_repo(mocker, implementation="gpu_cupy"),
            "mocked_module_path",
            "fake_method",
            MPI.COMM_WORLD,
        )

    assert "GPU is not available" in str(e)


def test_wrapper_passes_communicator_if_needed(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, comm: Optional[MPI.Comm] = None):
            assert comm is not None
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.execute(dummy_dataset)


def test_wrapper_allows_parameters_with_defaults(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, defaultpar: int = 10):
            assert defaultpar == 10
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.execute(dummy_dataset)


@pytest.mark.parametrize("enabled", [True, False])
def test_wrapper_processes_global_stats(
    mocker: MockerFixture, dummy_dataset: DataSet, enabled: bool
):
    stats_mock = mocker.patch(
        "httomo.runner.backend_wrapper.min_max_mean_std",
        return_value=(1.1, 2.2, 3.3, 4.4),
    )

    class FakeModule:
        def fake_method(data, glob_stats=None):
            if enabled:
                assert glob_stats == (1.1, 2.2, 3.3, 4.4)
            else:
                assert glob_stats is None
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "fake_method",
        MPI.COMM_WORLD,
        glob_stats=enabled,
    )

    wrp.execute(dummy_dataset)

    if enabled:
        stats_mock.assert_called_once()
    else:
        stats_mock.assert_not_called()


def test_wrapper_build_kwargs_parameter_not_given(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, param):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    with pytest.raises(ValueError) as e:
        wrp.execute(dummy_dataset)

    assert "Cannot map method parameter param to a value" in str(e)


def test_wrapper_access_outputref_params(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def fake_method(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    m = make_test_method(mocker)
    m.get_side_output.return_value = {"somepar": 42}
    wrp["param"] = OutputRef(m, "somepar")

    wrp.execute(dummy_dataset)


def test_wrapper_access_outputref_params_kwargs(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, **kwargs):
            assert kwargs["param"] == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    m = make_test_method(mocker)
    m.get_side_output.return_value = {"somepar": 42}
    wrp["param"] = OutputRef(m, "somepar")

    wrp.execute(dummy_dataset)


def test_wrapper_different_data_parameter_name(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(array):
            np.testing.assert_array_equal(array, 42)
            return array

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    dummy_dataset.data[:] = 42
    wrp.execute(dummy_dataset)


def test_wrapper_for_method_with_kwargs(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def fake_method(data, param, **kwargs):
            assert param == 42.0
            assert kwargs == {"extra": 123}
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "fake_method",
        MPI.COMM_WORLD,
        param=42.0,
        extra=123,
    )

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
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
        param=42,
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
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
    )
    wrp["param"] = 42
    wrp.execute(dummy_dataset)

    assert wrp["param"] == 42


def test_wrapper_sets_config_params_setter_not_in_arguments(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
    )
    with pytest.raises(ValueError):
        wrp["param_not_existing"] = 42


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
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
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
        make_mock_repo(mocker),
        "mocked_module_path",
        "normalize_tester",
        MPI.COMM_WORLD,
    )
    dummy_dataset.unlock()
    dummy_dataset.data[:] = 1
    dummy_dataset.darks[:] = 2
    dummy_dataset.flats[:] = 3
    dummy_dataset.lock()

    wrp.execute(dummy_dataset)

    importmock.assert_called_once_with("mocked_module_path")

@pytest.mark.parametrize("block", [False, True])
def test_wrapper_handles_reconstruction_angle_reshape(
    mocker: MockerFixture, dummy_dataset: DataSet, block: bool
):
    class FakeModule:
        # we give the angles a different name on purpose
        def recon_tester(data, theta):
            np.testing.assert_array_equal(data, 1)
            np.testing.assert_array_equal(theta, 2)
            assert data.shape[0] == len(theta)
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
    )
    dummy_dataset.data[:] = 1
    dummy_dataset.unlock()
    dummy_dataset.angles[:] = 2
    dummy_dataset.lock()
    
    input = dummy_dataset.make_block(0, 0, 3) if block else dummy_dataset

    wrp.execute(input)


def test_wrapper_handles_reconstruction_axisswap(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def recon_tester(data, theta):
            return data.swapaxes(0, 1)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
    )
    dummy_dataset.data = np.ones((13, 14, 15), dtype=np.float32)
    res = wrp.execute(dummy_dataset)

    assert res.data.shape == (13, 14, 15)


def test_wrapper_gives_recon_algorithm(mocker: MockerFixture):
    class FakeModule:
        def recon_tester(data, algorithm, center):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "test.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
    )

    assert wrp.recon_algorithm is None
    wrp["algorithm"] = "testalgo"
    assert wrp.recon_algorithm == "testalgo"


def test_wrapper_gives_no_recon_algorithm_if_not_recon_method(mocker: MockerFixture):
    class FakeModule:
        def tester(data, algorithm):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "test.something",
        "tester",
        MPI.COMM_WORLD,
    )

    assert wrp.recon_algorithm is None
    wrp["algorithm"] = "testalgo"
    assert wrp.recon_algorithm is None


def test_wrapper_rotation_fails_with_projection_method(mocker: MockerFixture):
    class FakeModule:
        def rotation_tester(data, ind=None):
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    with pytest.raises(NotImplementedError):
        make_backend_wrapper(
            make_mock_repo(mocker, pattern=Pattern.projection),
            "mocked_module_path.rotation",
            "rotation_tester",
            MPI.COMM_WORLD,
        )


def test_wrapper_rotation_accumulates_blocks(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def rotation_tester(data, ind=None):
            assert data.ndim == 2  # for 1 slice only
            np.testing.assert_array_equal(data, dummy_dataset.data[:, 4, :])
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        output_mapping={"cor": "cor"},
    )
    normalize = mocker.patch.object(
        wrp, "normalize_sino", side_effect=lambda sino, flats, darks: sino
    )
    # generate varying numbers so the comparison above works
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    b1 = dummy_dataset.make_block(0, 0, dummy_dataset.shape[0] // 2)
    b2 = dummy_dataset.make_block(
        0, dummy_dataset.shape[0] // 2, dummy_dataset.shape[0] // 2
    )
    wrp.execute(b1)
    normalize.assert_not_called()
    wrp.execute(b2)
    normalize.assert_called_once()
    assert wrp.get_side_output() == {"cor": 42.0}


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("ind_par", ["mid", 2, None])
def test_wrapper_rotation_gathers_single_sino_slice(
    mocker: MockerFixture,
    dummy_dataset: DataSet,
    rank: int,
    ind_par: Union[str, int, None],
    gpu: bool
):
    class FakeModule:
        def rotation_tester(data, ind=None):
            assert rank == 0  # for rank 1, it shouldn't be called
            assert data.ndim == 2  # for 1 slice only
            assert ind == 0
            if ind_par == "mid" or ind_par is None:
                xp.testing.assert_array_equal(
                    dummy_dataset.data[:, (dummy_dataset.data.shape[1] - 1) // 2, :],
                    data,
                )
            else:
                xp.testing.assert_array_equal(dummy_dataset.data[:, ind_par, :], data)
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
    )
    if ind_par is not None:
        wrp["ind"] = ind_par
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    if gpu:
        dummy_dataset.to_gpu()
    mocker.patch.object(wrp, "_gather_sino_slice", side_effect=lambda s: wrp.sino)
    normalize = mocker.patch.object(
        wrp, "normalize_sino", side_effect=lambda sino, f, d: sino
    )
    comm = mocker.patch.object(wrp, "comm")
    comm.rank = rank
    comm.size = 2
    comm.bcast.return_value = 42.0

    res = wrp.execute(dummy_dataset)

    assert wrp.pattern == Pattern.projection
    xp.testing.assert_array_equal(res.data, dummy_dataset.data)
    comm.bcast.assert_called_once()
    if rank == 0:
        normalize.assert_called_once()
    else:
        normalize.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_wrapper_gather_sino_slice(mocker: MockerFixture, rank: int):
    class FakeModule:
        def rotation_tester(data, ind=None):
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp: RotationWrapper = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
    )
    if rank == 0:
        wrp.sino = np.arange(2 * 6, dtype=np.float32).reshape((2, 6))
    else:
        wrp.sino = np.arange(2 * 6, 5 * 6, dtype=np.float32).reshape((3, 6))
    comm = mocker.patch.object(wrp, "comm")
    comm.rank = rank
    comm.size = 2
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


def test_wrapper_rotation_normalize_sino_no_darks_flats():
    ret = RotationWrapper.normalize_sino(
        np.ones((10, 10), dtype=np.float32), None, None
    )

    assert ret.shape == (10, 1, 10)
    np.testing.assert_allclose(np.squeeze(ret), 1.0)


def test_wrapper_rotation_normalize_sino_same_darks_flats():
    ret = RotationWrapper.normalize_sino(
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


def test_wrapper_rotation_normalize_sino_scalar():
    ret = RotationWrapper.normalize_sino(
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


def test_wrapper_rotation_normalize_sino_different_darks_flats():
    ret = RotationWrapper.normalize_sino(
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
def test_wrapper_rotation_normalize_sino_different_darks_flats_gpu():
    ret = RotationWrapper.normalize_sino(
        2.0 * xp.ones((10, 10), dtype=np.float32),
        1.0 * xp.ones((10, 10), dtype=np.float32),
        0.5 * xp.ones((10, 10), dtype=np.float32),
    )

    assert ret.shape == (10, 1, 10)
    assert getattr(ret, "device", None) is not None
    xp.testing.assert_allclose(xp.squeeze(ret), 1.0)



def test_wrapper_rotation_180(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def rotation_tester(data, ind):
            return 42.0  # center of rotation

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        output_mapping={"cor": "center"},
        ind=5,
    )

    new_dataset = wrp.execute(dummy_dataset)

    assert wrp.get_side_output() == {"center": 42.0}
    assert new_dataset == dummy_dataset  # note: not a deep comparison


def test_wrapper_rotation_360(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def rotation_tester(data, ind):
            # cor, overlap, side, overlap_position - from find_center_360
            return 42.0, 3.0, 1, 10.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
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
    new_dataset = wrp.execute(dummy_dataset)

    assert wrp.get_side_output() == {
        "center": 42.0,
        "overlap": 3.0,
        "pos": 10.0,
    }
    assert new_dataset == dummy_dataset  # note: not a deep comparison


def test_wrapper_dezinging(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def remove_outlier3d(x):
            return 2 * x

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.prep",
        "remove_outlier3d",
        MPI.COMM_WORLD,
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

    # it should only update the darks/flats once, but still apply to the data
    newset = wrp.execute(newset)
    np.testing.assert_array_equal(newset.data, 4)
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
        make_mock_repo(mocker, implementation="cpu"),
        "mocked_module_path.images",
        "save_to_images",
        MPI.COMM_WORLD,
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
        def save_to_images(data, out_dir, comm_rank):
            assert getattr(data, "device", None) is None  # make sure it's on CPU

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.images",
        "save_to_images",
        MPI.COMM_WORLD,
    )
    with xp.cuda.Device(0):
        dummy_dataset.to_gpu()
        new_dataset = wrp.execute(dummy_dataset)

        assert new_dataset.is_gpu is True


@pytest.mark.parametrize(
    "implementation,is_cpu,is_gpu,cupyrun",
    [
        ("cpu", True, False, False),
        ("gpu", False, True, False),
        ("gpu_cupy", False, True, True),
    ],
)
def test_wrapper_method_queries(
    mocker: MockerFixture,
    implementation: str,
    is_cpu: bool,
    is_gpu: bool,
    cupyrun: bool,
):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu = [GpuMemoryRequirement(dataset="tomo", multiplier=1.2, method="direct")]
    wrp = make_backend_wrapper(
        make_mock_repo(
            mocker,
            pattern=Pattern.projection,
            output_dims_change=True,
            implementation=implementation,
            memory_gpu=memory_gpu,
        ),
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )

    assert wrp.pattern == Pattern.projection
    assert wrp.output_dims_change is True
    assert wrp.implementation == implementation
    assert wrp.is_cpu == is_cpu
    assert wrp.is_gpu == is_gpu
    assert wrp.cupyrun == cupyrun
    assert wrp.memory_gpu == memory_gpu


@pytest.mark.parametrize("implementation", ["cpu", "gpu", "gpu_cupy"])
@pytest.mark.parametrize(
    "memory_gpu",
    [[GpuMemoryRequirement(dataset="tomo", multiplier=2.0, method="direct")], 
     [GpuMemoryRequirement(dataset="tomo", multiplier=0.0, method="direct")],
     []],
)
def test_wrapper_calculate_max_slices_direct(
    mocker: MockerFixture,
    dummy_dataset: DataSet,
    implementation: str,
    memory_gpu: List[GpuMemoryRequirement],
):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    wrp = make_backend_wrapper(
        make_mock_repo(
            mocker,
            pattern=Pattern.projection,
            output_dims_change=True,
            implementation=implementation,
            memory_gpu=memory_gpu,
        ),
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )
    shape = list(dummy_dataset.data.shape)
    shape.pop(0)
    databytes = shape[0] * shape[1] * dummy_dataset.data.itemsize
    max_slices_expected = 5
    multiplier = memory_gpu[0].multiplier if memory_gpu != [] else 1
    available_memory_in = int(databytes * max_slices_expected * multiplier)
    if available_memory_in == 0:
        available_memory_in = 5
    max_slices, available_memory = wrp.calculate_max_slices(
        dummy_dataset, shape, available_memory_in
    )

    if gpu_enabled and implementation != "cpu":
        assert max_slices == max_slices_expected
    else:
        assert max_slices > dummy_dataset.data.shape[0]
    assert available_memory == available_memory_in


def test_wrapper_calculate_max_slices_direct_flats_darks(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    memory_gpu = [
        GpuMemoryRequirement(dataset="tomo", multiplier=2.0, method="direct"),
        GpuMemoryRequirement(dataset="flats", multiplier=1.1, method="direct"),
        GpuMemoryRequirement(dataset="darks", multiplier=1.2, method="direct"),
    ]
    wrp = make_backend_wrapper(
        make_mock_repo(
            mocker,
            pattern=Pattern.projection,
            output_dims_change=True,
            implementation="gpu_cupy",
            memory_gpu=memory_gpu,
        ),
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )
    shape = list(dummy_dataset.data.shape)
    shape.pop(0)
    databytes = shape[0] * shape[1] * dummy_dataset.data.itemsize
    max_slices_expected = 5
    multiplier = memory_gpu[0].multiplier
    available_memory_in = int(databytes * max_slices_expected * multiplier)
    available_memory_adjusted = (
        available_memory_in
        + dummy_dataset.flats.nbytes * memory_gpu[1].multiplier
        + dummy_dataset.darks.nbytes * memory_gpu[2].multiplier
    )
    max_slices, available_memory = wrp.calculate_max_slices(
        dummy_dataset, shape, available_memory_adjusted
    )

    if gpu_enabled:
        assert max_slices == max_slices_expected
    else:
        assert max_slices > dummy_dataset.data.shape[0]
    assert available_memory == available_memory_in


def test_wrapper_calculate_max_slices_module(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu = [
        GpuMemoryRequirement(dataset="tomo", multiplier=None, method="module")
    ]
    wrp = make_backend_wrapper(
        make_mock_repo(
            mocker,
            pattern=Pattern.projection,
            output_dims_change=True,
            implementation="gpu_cupy",
            memory_gpu=memory_gpu,
        ),
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )
    memcalc_mock = mocker.patch.object(
        wrp.query, "calculate_memory_bytes", return_value=(1234, 5678)
    )
    shape = list(dummy_dataset.data.shape)
    shape.pop(0)
    max_slices, available_memory = wrp.calculate_max_slices(
        dummy_dataset, tuple(shape), 1_000_000_000
    )

    if gpu_enabled:
        assert max_slices == (1_000_000_000 - 5678) // 1234
        assert available_memory == 1_000_000_000
        memcalc_mock.assert_called_once_with(tuple(shape), dummy_dataset.data.dtype)
    else:
        assert max_slices > dummy_dataset.data.shape[0]
        assert available_memory == 1_000_000_000


def test_wrapper_calculate_output_dims(mocker: MockerFixture):
    class FakeModule:
        def test_method(data, testparam):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu = []
    wrp = make_backend_wrapper(
        make_mock_repo(
            mocker,
            pattern=Pattern.projection,
            output_dims_change=True,
            implementation="gpu_cupy",
            memory_gpu=memory_gpu,
        ),
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )
    wrp["testparam"] = 32

    memcalc_mock = mocker.patch.object(
        wrp.query, "calculate_output_dims", return_value=(1234, 5678)
    )

    dims = wrp.calculate_output_dims((10, 10))

    assert dims == (1234, 5678)
    memcalc_mock.assert_called_with((10, 10), testparam=32)


def test_wrapper_calculate_output_dims_no_change(mocker: MockerFixture):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu = []
    wrp = make_backend_wrapper(
        make_mock_repo(
            mocker,
            pattern=Pattern.projection,
            output_dims_change=False,
            implementation="gpu_cupy",
            memory_gpu=memory_gpu,
        ),
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )

    dims = wrp.calculate_output_dims((10, 10))

    assert dims == (10, 10)
