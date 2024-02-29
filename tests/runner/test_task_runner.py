from os import PathLike
from typing import List
from unittest.mock import ANY, call
import pytest
import numpy as np
from pytest_mock import MockerFixture
import httomo
from httomo.data.dataset_store import DataSetStoreWriter
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.pipeline import Pipeline
from httomo.runner.section import Section, sectionize
from httomo.runner.task_runner import TaskRunner
from httomo.utils import Pattern, xp, gpu_enabled
from httomo.runner.method_wrapper import MethodWrapper
from ..testing_utils import make_test_loader, make_test_method


def test_check_params_for_sweep_raises_exception(
    mocker: MockerFixture, tmp_path: PathLike
):
    p = Pipeline(
        loader=make_test_loader(mocker),
        methods=[
            make_test_method(
                mocker,
                testparam=(
                    1,
                    2,
                ),
            )
        ],
    )
    t = TaskRunner(p, reslice_dir=tmp_path)
    with pytest.raises(ValueError) as e:
        t._check_params_for_sweep()


def test_can_load_datasets(mocker: MockerFixture, tmp_path: PathLike):
    loader = make_test_loader(mocker)
    mksrc = mocker.patch.object(loader, "make_data_source")
    p = Pipeline(loader=loader, methods=[make_test_method(mocker)])
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()

    mksrc.assert_called()
    assert t.source is not None


def test_can_determine_max_slices_no_gpu_estimator(
    mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
):
    loader = make_test_loader(mocker, dummy_block)
    method = make_test_method(mocker, gpu=True, memory_gpu=[])
    p = Pipeline(loader=loader, methods=[method])
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(p)

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == dummy_block.chunk_shape[0]


@pytest.mark.parametrize("slices", [10, 500])
def test_can_determine_max_slices_empty_section(
    mocker: MockerFixture, tmp_path: PathLike, slices: int
):
    data = np.ones((slices, 10, 10), dtype=np.float32)
    aux = AuxiliaryData(angles=np.ones(slices, dtype=np.float32))
    block = DataSetBlock(data, aux)

    loader = make_test_loader(mocker, block)
    method = make_test_method(mocker, gpu=True, memory_gpu=[])
    p = Pipeline(loader=loader, methods=[method])
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(p)
    s.insert(0, Section(pattern=Pattern.sinogram, max_slices=0, methods=[]))

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == min(slices, httomo.globals.MAX_CPU_SLICES)


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.parametrize(
    "max_slices_methods",
    [
        [3],
        [5, 3, 19],
    ],
)
def test_can_determine_max_slices_with_gpu_estimator(
    mocker: MockerFixture,
    max_slices_methods: List[int],
    tmp_path: PathLike,
    dummy_block: DataSetBlock,
):
    loader = make_test_loader(mocker, dummy_block)
    methods: List[MethodWrapper] = []
    calc_dims_mocks = []
    calc_max_slices_mocks = []
    for i, max_slices in enumerate(max_slices_methods):
        method = make_test_method(mocker, gpu=True)
        mocker.patch.object(
            method,
            "memory_gpu",
            [GpuMemoryRequirement(dataset="tomo", multiplier=2.0, method="direct")],
        )
        calc_dims_mocks.append(
            mocker.patch.object(method, "calculate_output_dims", return_value=(10, 10))
        )
        calc_max_slices_mocks.append(
            mocker.patch.object(
                method, "calculate_max_slices", return_value=(max_slices, 1000000)
            )
        )
        methods.append(method)
    p = Pipeline(loader=loader, methods=methods)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(
        p,
    )
    assert len(s[0]) == len(max_slices_methods)
    shape = (dummy_block.shape[1], dummy_block.shape[2])

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == min(max_slices_methods)
    for i in range(len(max_slices_methods)):
        calc_dims_mocks[i].assert_called_with(shape)
        calc_max_slices_mocks[i].assert_called_with(
            dummy_block.data.dtype,
            shape,
            ANY,
            dummy_block.darks,
            dummy_block.flats,
        )


def test_can_determine_max_slices_with_cpu(
    mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
):
    loader = make_test_loader(mocker, dummy_block)
    methods = []
    for _ in range(3):
        method = make_test_method(mocker, gpu=False)
        methods.append(method)
    p = Pipeline(loader=loader, methods=methods)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(p)

    t.determine_max_slices(s[0], 0)
    assert s[0].max_slices == dummy_block.chunk_shape[0]


def test_can_determine_max_slices_with_cpu_large(
    mocker: MockerFixture, tmp_path: PathLike
):
    mocker.patch.object(httomo.globals, "MAX_CPU_SLICES", 16)
    data = np.ones((500, 10, 10), dtype=np.float32)
    aux = AuxiliaryData(angles=np.ones(500, dtype=np.float32))
    block = DataSetBlock(data, aux)
    loader = make_test_loader(mocker, block)
    methods = []
    for _ in range(3):
        method = make_test_method(mocker, gpu=False)
        methods.append(method)
    p = Pipeline(loader=loader, methods=methods)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(p)

    t.determine_max_slices(s[0], 0)
    assert s[0].max_slices == 16


# def test_calls_append_side_outputs_after_last_block(
#     mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
# ):
#     loader = make_test_loader(mocker, dummy_block)

#     side_outputs = {"answer": 42, "other": "xxx"}

#     block1 = dummy_dataset.make_block(0, 0, dummy_dataset.shape[0] // 2)
#     block2 = dummy_dataset.make_block(0, dummy_dataset.shape[0] // 2)
#     method1 = make_test_method(mocker)
#     mocker.patch.object(method1, "execute", side_effect=[block1, block2])
#     getmock = mocker.patch.object(method1, "get_side_output", return_value=side_outputs)
#     method2 = make_test_method(mocker)

#     p = Pipeline(loader=loader, methods=[method1, method2])
#     t = TaskRunner(p, reslice_dir=tmp_path)
#     spy = mocker.patch.object(t, "append_side_outputs")
#     t._prepare()
#     t._execute_method(method1, block1)
#     t._execute_method(method1, block2)  # this should trigger it

#     getmock.assert_called_once()
#     spy.assert_called_once_with(side_outputs)
#     t.side_outputs == side_outputs


def test_update_side_inputs_updates_downstream_methods(
    mocker: MockerFixture, tmp_path: PathLike
):
    loader = make_test_loader(mocker)
    side_outputs = {"answer": 42, "other": "xxx"}
    method1 = make_test_method(mocker, method_name="m1")
    method2 = make_test_method(mocker, method_name="m2")
    mocker.patch.object(method2, "parameters", ["answer"])
    setitem2 = mocker.patch.object(method2, "__setitem__")
    method3 = make_test_method(mocker, method_name="m3")
    mocker.patch.object(method3, "parameters", ["answer", "other", "whatever"])
    setitem3 = mocker.patch.object(method3, "__setitem__")

    p = Pipeline(loader=loader, methods=[method1, method2, method3])
    t = TaskRunner(p, reslice_dir=tmp_path)
    t.side_outputs = side_outputs
    t.set_side_inputs(method2)
    t.set_side_inputs(method3)

    setitem2.assert_called_with("answer", 42)
    method3_calls = [call("answer", 42), call("other", "xxx")]
    setitem3.assert_has_calls(method3_calls)


def test_execute_section_calls_blockwise_execute(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: PathLike
):
    original_value = dummy_block.data[0, 0, 0]  # it has all the same number
    loader = make_test_loader(mocker, dummy_block)
    method = make_test_method(mocker, method_name="m1")
    p = Pipeline(loader=loader, methods=[method])
    s = sectionize(p)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    # make that do nothing
    mocker.patch.object(t, "determine_max_slices")
    s[0].max_slices = dummy_block.chunk_shape[0] // 2  # we'll have 2 blocks

    # make this function return the block, with data multiplied by 2
    def mul_block_by_two(section, block: DataSetBlock):
        block.data *= 2
        return block

    block_mock = mocker.patch.object(
        t, "_execute_section_block", side_effect=mul_block_by_two
    )

    s[0].is_last = False  # should setup a DataSetStoreWriter as sink
    t._execute_section(s[0], 1)
    assert isinstance(t.sink, DataSetStoreWriter)
    reader = t.sink.make_reader()
    data = reader.read_block(0, dummy_block.shape[0])

    np.testing.assert_allclose(data.data, original_value * 2)
    calls = [call(ANY, ANY), call(ANY, ANY)]
    block_mock.assert_has_calls(calls)  # make sure we got called twice


def test_execute_section_for_block(
    mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
):
    loader = make_test_loader(mocker, dummy_block)
    method1 = make_test_method(mocker, method_name="m1")
    method2 = make_test_method(mocker, method_name="m2")
    p = Pipeline(loader=loader, methods=[method1, method2])
    s = sectionize(p)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    exec_method = mocker.patch.object(t, "_execute_method", return_value=dummy_block)
    t._execute_section_block(s[0], dummy_block)

    calls = [call(method1, ANY), call(method2, ANY)]
    exec_method.assert_has_calls(calls)


def test_does_reslice_when_needed(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: PathLike
):
    loader = make_test_loader(mocker, dummy_block)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    mocker.patch.object(method1, "execute", return_value=dummy_block)
    method2 = make_test_method(mocker, method_name="m2", pattern=Pattern.sinogram)
    # create a block in the other slicing dim
    block2 = DataSetBlock(data=dummy_block.data, 
                          aux_data=dummy_block.aux_data, 
                          slicing_dim=1)
    mocker.patch.object(method2, "execute", return_value=block2)
    p = Pipeline(loader=loader, methods=[method1, method2])
    t = TaskRunner(p, reslice_dir=tmp_path)

    t.execute()

    assert loader.pattern == Pattern.projection
    assert t.source is not None
    assert t.sink is not None
    assert t.source.slicing_dim == 1
    assert t.sink.slicing_dim == 1


@pytest.mark.parametrize(
    "loader_pattern,reslices",
    [(Pattern.all, 2), (Pattern.projection, 2), (Pattern.sinogram, 3)],
)
def test_warns_with_multiple_reslices(
    mocker: MockerFixture,
    dummy_block: DataSetBlock,
    tmp_path: PathLike,
    loader_pattern: Pattern,
    reslices: int,
):
    loader = make_test_loader(mocker, dummy_block, pattern=loader_pattern)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    method2 = make_test_method(mocker, method_name="m2", pattern=Pattern.sinogram)
    method3 = make_test_method(mocker, method_name="m3", pattern=Pattern.projection)
    p = Pipeline(loader=loader, methods=[method1, method2, method3])
    t = TaskRunner(p, reslice_dir=tmp_path)

    spy = mocker.patch("httomo.runner.task_runner.log_once")

    t._sectionize()

    spy.assert_called()
    args, kwargs = spy.call_args
    assert f"Reslicing will be performed {reslices} times" in args[0]
