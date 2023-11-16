from typing import List, Optional, Union
from mpi4py import MPI
import numpy as np
import httomo
from httomo.runner.dataset import DataSet
from httomo.runner.backend_wrapper import make_backend_wrapper
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.output_ref import OutputRef
from httomo.utils import Pattern, xp, gpu_enabled
from pytest_mock import MockerFixture
import pytest

from .testing_utils import make_mock_repo, make_test_method


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


def test_wrapper_handles_reconstruction_angle_reshape(
    mocker: MockerFixture, dummy_dataset: DataSet
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

    wrp.execute(dummy_dataset)


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


def test_wrapper_rotation_180(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def rotation_tester(data):
            return 42.0  # center of rotation

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
        output_mapping={"cor": "center"},
    )

    new_dataset = wrp.execute(dummy_dataset)

    assert wrp.get_side_output() == {"center": 42.0}
    assert new_dataset == dummy_dataset  # note: not a deep comparison


def test_wrapper_rotation_360(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def rotation_tester(data):
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
    )
    new_dataset = wrp.execute(dummy_dataset)

    assert wrp.get_side_output() == {
        "center": 42.0,
        "overlap": 3.0,
        "pos": 10.0,
    }
    assert new_dataset == dummy_dataset  # note: not a deep comparison


@pytest.mark.parametrize("ind_par", ["mid", 2, None])
def test_wrapper_rotation_ind_parameter(
    mocker: MockerFixture, dummy_dataset: DataSet, ind_par: Union[str, int, None]
):
    class FakeModule:
        def rotation_tester(data, ind: int):
            if ind_par == "mid" or ind_par is None:
                assert ind == (dummy_dataset.data.shape[1] + 1) // 2
            else:
                assert ind == ind_par
            return 42.0

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_backend_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.rotation",
        "rotation_tester",
        MPI.COMM_WORLD,
    )
    if ind_par is not None:
        wrp["ind"] = ind_par
    wrp.execute(dummy_dataset)


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
    [[GpuMemoryRequirement(dataset="tomo", multiplier=2.0, method="direct")], []],
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
