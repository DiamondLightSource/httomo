from typing import List, Optional

import numpy as np
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.dataset import DataSet
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.output_ref import OutputRef
from httomo.utils import Pattern, gpu_enabled
from ..testing_utils import make_mock_repo, make_test_method

import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_generic_get_name_and_paths(mocker: MockerFixture):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "testmodule.path",
        "fake_method",
        MPI.COMM_WORLD,
    )
    assert isinstance(wrp, GenericMethodWrapper)
    assert wrp.method_name == "fake_method"
    assert wrp.module_path == "testmodule.path"
    assert wrp.package_name == "testmodule"
    assert wrp.task_id == ""
    assert wrp.save_result is False


def test_generic_set_task_id(mocker: MockerFixture):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "testmodule.path",
        "fake_method",
        MPI.COMM_WORLD,
        task_id="fake_method_id"
    )
    
    assert wrp.task_id == "fake_method_id"

def test_generic_execute_transfers_to_gpu(
    dummy_dataset: DataSet, mocker: MockerFixture
):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "module_path",
        "fake_method",
        MPI.COMM_WORLD,
    )
    dataset = wrp.execute(dummy_dataset.make_block(0))

    assert dataset.is_gpu == gpu_enabled


def test_generic_execute_calls_pre_post_process(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "module_path",
        "fake_method",
        MPI.COMM_WORLD,
    )
    block = dummy_dataset.make_block(0)
    prep = mocker.patch.object(wrp, "_preprocess_data", return_value=block)
    post = mocker.patch.object(wrp, "_postprocess_data", return_value=block)
    trans = mocker.patch.object(wrp, "_transfer_data", return_value=block)

    wrp.execute(block)

    prep.assert_called_once_with(block)
    post.assert_called_once_with(block)
    trans.assert_called_once_with(block)


def test_generic_fails_with_wrong_returntype(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data):
            # returning None should not be allowed for generic wrapper,
            # it must be a cpu or gpu array (data)
            return None

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    with pytest.raises(ValueError) as e:
        wrp.execute(dummy_dataset.make_block(0))
    assert "return type" in str(e)


def test_generic_sets_gpuid(mocker: MockerFixture, dummy_dataset: DataSet):
    mocker.patch("httomo.method_wrappers.generic.gpu_enabled", True)
    mocker.patch(
        "httomo.method_wrappers.generic.xp.cuda.runtime.getDeviceCount", return_value=4
    )
    mocker.patch("httomo.method_wrappers.generic.httomo.globals.gpu_id", -1)
    mocker.patch("httomo.method_wrappers.generic.mpiutil.local_rank", 3)

    class FakeModule:
        def fake_method(data, gpu_id: int):
            assert gpu_id == 3
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.execute(dummy_dataset.make_block(0))


def test_generic_fails_for_gpumethods_with_no_gpu(mocker: MockerFixture):
    mocker.patch("httomo.method_wrappers.generic.gpu_enabled", False)

    class FakeModule:
        def fake_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    with pytest.raises(ValueError) as e:
        make_method_wrapper(
            make_mock_repo(mocker, implementation="gpu_cupy"),
            "mocked_module_path",
            "fake_method",
            MPI.COMM_WORLD,
        )

    assert "GPU is not available" in str(e)


def test_generic_passes_communicator_if_needed(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, comm: Optional[MPI.Comm] = None):
            assert comm is not None
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.execute(dummy_dataset.make_block(0))


def test_generic_allows_parameters_with_defaults(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, defaultpar: int = 10):
            assert defaultpar == 10
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.execute(dummy_dataset.make_block(0))


def test_generic_build_kwargs_parameter_not_given(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, param):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    with pytest.raises(ValueError) as e:
        wrp.execute(dummy_dataset.make_block(0))

    assert "Cannot map method parameter param to a value" in str(e)


def test_generic_access_outputref_params(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def fake_method(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    m = make_test_method(mocker)
    mocker.patch.object(m, "get_side_output", return_value={"somepar": 42})
    wrp["param"] = OutputRef(m, "somepar")

    wrp.execute(dummy_dataset.make_block(0))


def test_generic_access_outputref_params_kwargs(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(data, **kwargs):
            assert kwargs["param"] == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    m = make_test_method(mocker)
    mocker.patch.object(m, "get_side_output", return_value={"somepar": 42})
    wrp["param"] = OutputRef(m, "somepar")

    wrp.execute(dummy_dataset.make_block(0))


def test_generic_different_data_parameter_name(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def fake_method(array):
            np.testing.assert_array_equal(array, 42)
            return array

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    dummy_dataset.data[:] = 42
    wrp.execute(dummy_dataset.make_block(0))


def test_generic_for_method_with_kwargs(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def fake_method(data, param, **kwargs):
            assert param == 42.0
            assert kwargs == {"extra": 123}
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "fake_method",
        MPI.COMM_WORLD,
        param=42.0,
        extra=123,
    )

    wrp.execute(dummy_dataset.make_block(0))


def test_generic_sets_config_params_constructor(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
        param=42,
    )
    wrp.execute(dummy_dataset.make_block(0))

    assert wrp["param"] == 42


def test_generic_sets_config_params_setter(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
    )
    wrp["param"] = 42
    wrp.execute(dummy_dataset.make_block(0))

    assert wrp["param"] == 42


def test_generic_sets_config_params_setter_not_in_arguments(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param):
            assert param == 42
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
    )
    with pytest.raises(ValueError):
        wrp["param_not_existing"] = 42


def test_generic_sets_config_params_append_dict(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def param_tester(data, param1, param2):
            assert param1 == 42
            assert param2 == 43
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "param_tester",
        MPI.COMM_WORLD,
    )
    wrp.append_config_params({"param1": 42, "param2": 43})
    wrp.execute(dummy_dataset.make_block(0))

    assert wrp["param1"] == 42
    assert wrp["param2"] == 43


def test_generic_passes_darks_flats_to_normalize(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def normalize_tester(data, flats, darks):
            np.testing.assert_array_equal(data, 1)
            np.testing.assert_array_equal(darks, 2)
            np.testing.assert_array_equal(flats, 3)
            return data

    importmock = mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
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

    wrp.execute(dummy_dataset.make_block(0))

    importmock.assert_called_once_with("mocked_module_path")


@pytest.mark.parametrize(
    "implementation,is_cpu,is_gpu,cupyrun",
    [
        ("cpu", True, False, False),
        ("gpu", False, True, False),
        ("gpu_cupy", False, True, True),
    ],
)
def test_generic_method_queries(
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
    wrp = make_method_wrapper(
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
    [
        [GpuMemoryRequirement(dataset="tomo", multiplier=2.0, method="direct")],
        [GpuMemoryRequirement(dataset="tomo", multiplier=0.0, method="direct")],
        [],
    ],
)
def test_generic_calculate_max_slices_direct(
    mocker: MockerFixture,
    dummy_dataset: DataSet,
    implementation: str,
    memory_gpu: List[GpuMemoryRequirement],
):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    wrp = make_method_wrapper(
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
    shape_t = list(dummy_dataset.data.shape)
    shape_t.pop(0)
    shape = (shape_t[0], shape_t[1])
    databytes = shape[0] * shape[1] * dummy_dataset.data.itemsize
    max_slices_expected = 5
    multiplier = memory_gpu[0].multiplier if memory_gpu != [] else 1
    available_memory_in = int(databytes * max_slices_expected * multiplier)
    if available_memory_in == 0:
        available_memory_in = 5
    max_slices, available_memory = wrp.calculate_max_slices(
        dummy_dataset.data.dtype,
        shape,
        available_memory_in,
        dummy_dataset.darks,
        dummy_dataset.flats,
    )

    if gpu_enabled and implementation != "cpu":
        assert max_slices == max_slices_expected
    else:
        assert max_slices > dummy_dataset.data.shape[0]
    assert available_memory == available_memory_in


def test_generic_calculate_max_slices_direct_flats_darks(
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
    wrp = make_method_wrapper(
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
    shape_t = list(dummy_dataset.data.shape)
    shape_t.pop(0)
    shape = (shape_t[0], shape_t[1])
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
        dummy_dataset.data.dtype,
        shape,
        available_memory_adjusted,
        dummy_dataset.darks,
        dummy_dataset.flats,
    )

    if gpu_enabled:
        assert max_slices == max_slices_expected
    else:
        assert max_slices > dummy_dataset.data.shape[0]
    assert available_memory == available_memory_in


def test_generic_calculate_max_slices_module(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu = [
        GpuMemoryRequirement(dataset="tomo", multiplier=None, method="module")
    ]
    repo = make_mock_repo(
        mocker,
        pattern=Pattern.projection,
        output_dims_change=True,
        implementation="gpu_cupy",
        memory_gpu=memory_gpu,
    )
    memcalc_mock = mocker.patch.object(
        repo.query("", ""), "calculate_memory_bytes", return_value=(1234, 5678)
    )
    wrp = make_method_wrapper(
        repo,
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )
    shape_t = list(dummy_dataset.data.shape)
    shape_t.pop(0)
    shape = (shape_t[0], shape_t[1])
    max_slices, available_memory = wrp.calculate_max_slices(
        dummy_dataset.data.dtype,
        shape,
        1_000_000_000,
        dummy_dataset.darks,
        dummy_dataset.flats,
    )

    if gpu_enabled:
        assert max_slices == (1_000_000_000 - 5678) // 1234
        assert available_memory == 1_000_000_000
        memcalc_mock.assert_called_once_with(tuple(shape), dummy_dataset.data.dtype,
                                             darks_shape=dummy_dataset.darks.shape, 
                                             flats_shape=dummy_dataset.flats.shape, 
                                             darks_dtype=dummy_dataset.darks.dtype, 
                                             flats_dtype=dummy_dataset.flats.dtype)
    else:
        assert max_slices > dummy_dataset.data.shape[0]
        assert available_memory == 1_000_000_000


def test_generic_calculate_output_dims(mocker: MockerFixture):
    class FakeModule:
        def test_method(data, testparam):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu: List[GpuMemoryRequirement] = []
    repo = make_mock_repo(
        mocker,
        pattern=Pattern.projection,
        output_dims_change=True,
        implementation="gpu_cupy",
        memory_gpu=memory_gpu,
    )
    memcalc_mock = mocker.patch.object(
        repo.query("", ""), "calculate_output_dims", return_value=(1234, 5678)
    )
    wrp = make_method_wrapper(
        repo,
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )
    wrp["testparam"] = 32

    dims = wrp.calculate_output_dims((10, 10))

    assert dims == (1234, 5678)
    memcalc_mock.assert_called_with((10, 10), testparam=32)


def test_generic_calculate_output_dims_no_change(mocker: MockerFixture):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu: List[GpuMemoryRequirement] = []
    wrp = make_method_wrapper(
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
