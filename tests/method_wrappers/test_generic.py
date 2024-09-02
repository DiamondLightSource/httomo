from typing import List, Optional, Union

import numpy as np
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.dataset import DataSetBlock
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.output_ref import OutputRef
from httomo.utils import Pattern, gpu_enabled, xp
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
        make_mock_repo(mocker, padding=True),
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
    assert wrp.padding is True


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
        task_id="fake_method_id",
    )

    assert wrp.task_id == "fake_method_id"


@pytest.mark.cupy
def test_generic_execute_transfers_to_gpu(
    mocker: MockerFixture, dummy_block: DataSetBlock
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
    res = wrp.execute(dummy_block)

    assert res.is_gpu == gpu_enabled


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.cupy
def test_generic_excute_measures_gpu_times(
    dummy_block: DataSetBlock, mocker: MockerFixture
):
    class FakeModule:
        def fake_method(data):
            return xp.exp(data) + data  # make sure some kernel code is run

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "module_path",
        "fake_method",
        MPI.COMM_WORLD,
    )
    wrp.execute(dummy_block)

    if gpu_enabled:
        assert wrp.gpu_time.host2device > 0.0
        assert wrp.gpu_time.device2host == 0.0
        assert wrp.gpu_time.kernel > 0.0


@pytest.mark.cupy
def test_generic_execute_calls_pre_post_process(
    mocker: MockerFixture, dummy_block: DataSetBlock
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
    prep = mocker.patch.object(wrp, "_preprocess_data", return_value=dummy_block)
    post = mocker.patch.object(wrp, "_postprocess_data", return_value=dummy_block)
    trans = mocker.patch.object(wrp, "_transfer_data", return_value=dummy_block)

    wrp.execute(dummy_block)

    prep.assert_called_once_with(dummy_block)
    post.assert_called_once_with(dummy_block)
    trans.assert_called_once_with(dummy_block)


def test_generic_fails_with_wrong_returntype(
    mocker: MockerFixture, dummy_block: DataSetBlock
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
        wrp.execute(dummy_block)
    assert "return type" in str(e)


@pytest.mark.cupy
def test_generic_sets_gpuid(mocker: MockerFixture, dummy_block: DataSetBlock):
    mocker.patch("httomo.method_wrappers.generic.gpu_enabled", True)
    mocker.patch("httomo.method_wrappers.generic.httomo.globals.gpu_id", -1)
    GPU_ID = 3
    gpu_id_getter_spy = mocker.patch(
        "httomo.method_wrappers.generic.get_gpu_id",
        return_value=GPU_ID,
    )

    class FakeModule:
        def fake_method(data, gpu_id: int):
            assert gpu_id == GPU_ID
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    gpu_id_getter_spy.assert_called_once()

    wrp.execute(dummy_block)


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
    mocker: MockerFixture, dummy_block: DataSetBlock
):
    class FakeModule:
        def fake_method(data, comm: Optional[MPI.Comm] = None):
            assert comm is not None
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.execute(dummy_block)


def test_generic_transforms_auto_axis(mocker: MockerFixture, dummy_block: DataSetBlock):
    PATTERN = Pattern.projection

    class FakeModule:
        def fake_method(data, axis: int):
            assert axis == PATTERN.value
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=PATTERN),
        "mocked_module_path",
        "fake_method",
        MPI.COMM_WORLD,
        axis="auto",
    )

    wrp.execute(dummy_block)


def test_generic_allows_parameters_with_defaults(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
    class FakeModule:
        def fake_method(data, defaultpar: int = 10):
            assert defaultpar == 10
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )

    wrp.execute(dummy_block)


def test_generic_build_kwargs_parameter_not_given(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
    class FakeModule:
        def fake_method(data, param):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    with pytest.raises(ValueError) as e:
        wrp.execute(dummy_block)

    assert "Cannot map method parameter param to a value" in str(e)


def test_generic_access_outputref_params(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
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

    wrp.execute(dummy_block)


def test_generic_access_outputref_params_kwargs(
    mocker: MockerFixture, dummy_block: DataSetBlock
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

    wrp.execute(dummy_block)


def test_generic_different_data_parameter_name(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
    class FakeModule:
        def fake_method(array):
            np.testing.assert_array_equal(array, 42)
            return array

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker), "mocked_module_path", "fake_method", MPI.COMM_WORLD
    )
    dummy_block.data[:] = 42
    wrp.execute(dummy_block)


def test_generic_for_method_with_kwargs(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
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

    wrp.execute(dummy_block)


def test_generic_sets_config_params_constructor(
    mocker: MockerFixture, dummy_block: DataSetBlock
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
    wrp.execute(dummy_block)

    assert wrp["param"] == 42


def test_generic_sets_config_params_setter(
    mocker: MockerFixture, dummy_block: DataSetBlock
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
    wrp.execute(dummy_block)

    assert wrp["param"] == 42


def test_generic_sets_config_params_setter_not_in_arguments(
    mocker: MockerFixture, dummy_block: DataSetBlock
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
    mocker: MockerFixture, dummy_block: DataSetBlock
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
    wrp.execute(dummy_block)

    assert wrp["param1"] == 42
    assert wrp["param2"] == 43


def test_generic_passes_darks_flats_to_normalize(
    mocker: MockerFixture, dummy_block: DataSetBlock
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
    dummy_block.data[:] = 1
    dummy_block.darks[:] = 2
    dummy_block.flats[:] = 3

    wrp.execute(dummy_block)

    importmock.assert_called_once_with("mocked_module_path")


@pytest.mark.parametrize(
    "implementation,is_cpu,is_gpu,cupyrun",
    [
        ("cpu", True, False, False),
        ("gpu", False, True, False),
        ("gpu_cupy", False, True, True),
    ],
)
@pytest.mark.cupy
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

    memory_gpu = GpuMemoryRequirement(multiplier=1.2, method="direct")
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
        GpuMemoryRequirement(multiplier=2.0, method="direct"),
        GpuMemoryRequirement(multiplier=0.0, method="direct"),
        None,
    ],
)
@pytest.mark.cupy
def test_generic_calculate_max_slices_direct(
    mocker: MockerFixture,
    dummy_block: DataSetBlock,
    implementation: str,
    memory_gpu: GpuMemoryRequirement,
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
    shape_t = list(dummy_block.chunk_shape)
    shape_t.pop(0)
    shape = (shape_t[0], shape_t[1])
    databytes = shape[0] * shape[1] * dummy_block.data.itemsize
    max_slices_expected = 5
    multiplier = float(
        memory_gpu.multiplier
        if memory_gpu is not None and memory_gpu.multiplier is not None
        else 1
    )
    available_memory_in = int(databytes * max_slices_expected * multiplier)
    if available_memory_in == 0:
        available_memory_in = 5
    max_slices, available_memory = wrp.calculate_max_slices(
        dummy_block.data.dtype,
        shape,
        available_memory_in,
    )

    if gpu_enabled and implementation != "cpu":
        assert max_slices == max_slices_expected
    else:
        assert max_slices > dummy_block.data.shape[0]
    assert available_memory == available_memory_in


@pytest.mark.cupy
def test_generic_calculate_max_slices_module(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu = GpuMemoryRequirement(multiplier=None, method="module")
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
    shape_t = list(dummy_block.chunk_shape)
    shape_t.pop(0)
    shape = (shape_t[0], shape_t[1])
    max_slices, available_memory = wrp.calculate_max_slices(
        dummy_block.data.dtype,
        shape,
        1_000_000_000,
    )

    if gpu_enabled:
        assert max_slices == (1_000_000_000 - 5678) // 1234
        assert available_memory == 1_000_000_000
        memcalc_mock.assert_called_once_with(tuple(shape), dummy_block.data.dtype)
    else:
        assert max_slices > dummy_block.chunk_shape[0]
        assert available_memory == 1_000_000_000


@pytest.mark.cupy
def test_generic_calculate_output_dims(mocker: MockerFixture):
    class FakeModule:
        def test_method(data, testparam):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu: GpuMemoryRequirement = None
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


@pytest.mark.cupy
def test_generic_calculate_output_dims_no_change(mocker: MockerFixture):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu: GpuMemoryRequirement = None
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


def test_generic_execute_uses_comm_passed_to_constructor(
    mocker: MockerFixture,
    dummy_block: DataSetBlock,
):
    # Define mock global comm, to be used in place of `MPI.COMM_WORLD` in the code to be tested
    global_comm_mock = mocker.create_autospec(MPI.Comm)
    global_comm_mock.size = 8
    global_comm_mock.rank = 0

    # Patch `httomo.runner.gpu_utils` import of `MPI.COMM_WORLD` to be the mock global
    # communicator object defined
    mocker.patch("httomo.runner.gpu_utils.MPI.COMM_WORLD", global_comm_mock)

    # Define spy on mock global communicator's `bcast()` method, for later assertions that its
    # `bcast()` method is never called.
    #
    # The fact that its `bcast()` method should never be called is used as a proxy/marker that
    # the global communicator is never used in method execution, only the communicator passed
    # into the method wrapper's constructor should be used in method execution.
    global_comm_bcast_spy = mocker.patch.object(
        target=global_comm_mock, attribute="bcast"
    )

    # Define mock comm object to pass into method wrapper constructor
    passed_in_mock_comm = mocker.create_autospec(MPI.Comm)
    passed_in_mock_comm.size = 1
    passed_in_mock_comm.rank = 0

    # Define spy on mock communicator passed into method wrapper constructor, for later
    # assertions that its `bcast()` method is only called when the method is executed
    passed_in_mock_comm_bcast_spy = mocker.patch.object(
        target=passed_in_mock_comm,
        attribute="bcast",
    )

    # Define dummy method function which has a `comm` parameter. This will make the method
    # wrapper pass in its `self.comm` to the method function. This communicator is the
    # communicator that is passed into the method wrapper constructor. Therefore, the
    # communicator in the method function *should* be the communicator that is passed into the
    # method wrapper's constructor, and *not* the global communicator.
    #
    # Suppose the communicator given to the method function is used in some manner (say, the
    # `bcast()` method is called) during the method execution. Then, a spy on the `bcast()`
    # method can be used to verify if the communicator being used for method execution is the
    # one passed into the method wrapper's constructor and *not* the global communicator.
    class FakeModule:
        def test_method(data, comm):
            comm.bcast(1, root=0)
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path",
        "test_method",
        comm=passed_in_mock_comm,
    )

    # Run `execute()` method, which should trigger the dummy method function defined earlier,
    # that will access the communicator given to it by the method wrapper (which will be the
    # communicator that was passed into the method wrapper constructor)
    #
    # The global communicator should never be executed, and the communicator passed into the
    # method wrapper constructor should only be executed once when the method is executed.
    global_comm_bcast_spy.assert_not_called()
    passed_in_mock_comm_bcast_spy.assert_not_called()
    wrp.execute(dummy_block)
    global_comm_bcast_spy.assert_not_called()
    passed_in_mock_comm_bcast_spy.assert_called_once()


@pytest.mark.cupy
def test_generic_calculate_padding_none_required(mocker: MockerFixture):
    class FakeModule:
        def test_method(data):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu: GpuMemoryRequirement = None
    wrp = make_method_wrapper(
        make_mock_repo(
            mocker,
            pattern=Pattern.projection,
            implementation="gpu_cupy",
            memory_gpu=memory_gpu,
            padding=False,
        ),
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )

    padding = wrp.calculate_padding()
    assert padding == (0, 0)


@pytest.mark.cupy
def test_generic_calculate_padding(mocker: MockerFixture):
    EXPECTED_PADDING = (5, 10)
    PARAM_AFFECTS_PADDING = 7

    class FakeModule:
        def test_method(data, param_affects_padding):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)

    memory_gpu: GpuMemoryRequirement = None
    repo = make_mock_repo(
        mocker,
        pattern=Pattern.projection,
        implementation="gpu_cupy",
        memory_gpu=memory_gpu,
        padding=True,
    )
    padding_calc_mock = mocker.patch.object(
        repo.query("", ""), "calculate_padding", return_value=EXPECTED_PADDING
    )
    wrp = make_method_wrapper(
        repo,
        "mocked_module_path",
        "test_method",
        MPI.COMM_WORLD,
    )
    wrp["param_affects_padding"] = PARAM_AFFECTS_PADDING

    padding = wrp.calculate_padding()

    assert padding == EXPECTED_PADDING
    padding_calc_mock.assert_called_with(param_affects_padding=PARAM_AFFECTS_PADDING)
