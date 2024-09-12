from os import PathLike
from typing import List, Tuple
from unittest.mock import ANY, call

import pytest
import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.data.dataset_store import DataSetStoreWriter
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_backing import DataSetStoreBacking
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.monitoring_interface import MonitoringInterface
from httomo.runner.output_ref import OutputRef
from httomo.runner.pipeline import Pipeline
from httomo.runner.section import Section, sectionize
from httomo.runner.task_runner import TaskRunner
from httomo.utils import (
    Pattern,
    xp,
    gpu_enabled,
)
from httomo.runner.method_wrapper import GpuTimeInfo, MethodWrapper
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
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError) as e:
        t._check_params_for_sweep()


def test_can_load_datasets(mocker: MockerFixture, tmp_path: PathLike):
    loader = make_test_loader(mocker)
    mksrc = mocker.patch.object(loader, "make_data_source")
    p = Pipeline(loader=loader, methods=[make_test_method(mocker)])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t._prepare()

    mksrc.assert_called()
    assert t.source is not None


def test_can_determine_max_slices_no_gpu_estimator(
    mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
):
    loader = make_test_loader(mocker, dummy_block)
    method = make_test_method(mocker, gpu=True, memory_gpu=None)
    p = Pipeline(loader=loader, methods=[method])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t._prepare()
    s = sectionize(p)

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == dummy_block.chunk_shape[0]


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
            GpuMemoryRequirement(multiplier=2.0, method="direct"),
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
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
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
        )


@pytest.mark.parametrize("limit", [0, 10000])
def test_can_determine_max_slices_with_gpu_estimator_and_cpu_limit(
    mocker: MockerFixture,
    limit: int,
    tmp_path: PathLike,
    dummy_block: DataSetBlock,
):
    mocker.patch("httomo.runner.task_runner.get_available_gpu_memory", return_value=1e7)
    loader = make_test_loader(mocker, dummy_block)
    methods: List[MethodWrapper] = []
    calc_dims_mocks = []
    calc_max_slices_mocks = []
    for i in range(3):
        method = make_test_method(mocker, gpu=True)
        mocker.patch.object(
            method,
            "memory_gpu",
            GpuMemoryRequirement(multiplier=2.0, method="direct"),
        )
        calc_dims_mocks.append(
            mocker.patch.object(method, "calculate_output_dims", return_value=(10, 10))
        )
        calc_max_slices_mocks.append(
            mocker.patch.object(
                method,
                "calculate_max_slices",
                return_value=(10, limit if limit != 0 else 1e7),
            )
        )
        methods.append(method)
    p = Pipeline(loader=loader, methods=methods)

    t = TaskRunner(
        p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD, memory_limit_bytes=limit
    )
    t._prepare()
    s = sectionize(p)
    shape = (dummy_block.shape[1], dummy_block.shape[2])

    t.determine_max_slices(s[0], 0)

    for i in range(3):
        calc_dims_mocks[i].assert_called_with(shape)
        calc_max_slices_mocks[i].assert_called_with(
            dummy_block.data.dtype,
            shape,
            limit if limit != 0 else 1e7,
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
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t._prepare()
    s = sectionize(p)

    t.determine_max_slices(s[0], 0)
    assert s[0].max_slices == dummy_block.chunk_shape[0]


def test_can_determine_max_slices_with_cpu_large(
    mocker: MockerFixture, tmp_path: PathLike
):
    data = np.ones((500, 10, 10), dtype=np.float32)
    aux = AuxiliaryData(angles=np.ones(500, dtype=np.float32))
    block = DataSetBlock(data, aux)
    loader = make_test_loader(mocker, block)
    methods = []
    for _ in range(3):
        method = make_test_method(mocker, gpu=False)
        methods.append(method)
    p = Pipeline(loader=loader, methods=methods)
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t._prepare()
    s = sectionize(p)

    t.determine_max_slices(s[0], 0)
    assert s[0].max_slices == 64


def test_append_side_outputs(mocker: MockerFixture, tmp_path: PathLike):
    p = Pipeline(make_test_loader(mocker), methods=[])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t.append_side_outputs({"answer": 42.0, "other": "xxx"})
    assert t.side_outputs == {"answer": 42.0, "other": "xxx"}


def test_calls_append_side_outputs_after_last_block(
    mocker: MockerFixture,
    tmp_path: PathLike,
):
    GLOBAL_SHAPE = (500, 10, 10)
    CHUNK_SHAPE = GLOBAL_SHAPE
    data = np.ones(GLOBAL_SHAPE, dtype=np.float32)
    aux = AuxiliaryData(angles=np.ones(GLOBAL_SHAPE[0], dtype=np.float32))
    block1 = DataSetBlock(
        data=data[: GLOBAL_SHAPE[0] // 2, :, :],
        aux_data=aux,
        block_start=0,
        chunk_start=0,
        chunk_shape=CHUNK_SHAPE,
        global_shape=GLOBAL_SHAPE,
    )
    block2 = DataSetBlock(
        data=data[GLOBAL_SHAPE[0] // 2 :, :, :],
        aux_data=aux,
        block_start=CHUNK_SHAPE[0] // 2,
        chunk_start=0,
        chunk_shape=CHUNK_SHAPE,
        global_shape=GLOBAL_SHAPE,
    )

    method = make_test_method(mocker)
    mocker.patch.object(method, "execute", side_effect=[block1, block2])
    side_outputs = {"answer": 42, "other": "xxx"}
    getmock = mocker.patch.object(method, "get_side_output", return_value=side_outputs)

    loader = make_test_loader(mocker)
    p = Pipeline(loader=loader, methods=[method])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    spy = mocker.patch.object(t, "append_side_outputs")
    t._prepare()
    t._execute_method(
        method, block1
    )  # the first block shouldn't trigger a side output append call
    assert spy.call_count == 0

    t._execute_method(
        method, block2
    )  # the last block should trigger side output append call
    getmock.assert_called_once()
    spy.assert_called_once_with(side_outputs)


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
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t.side_outputs = side_outputs
    t.set_side_inputs(method2)
    t.set_side_inputs(method3)

    setitem2.assert_called_with("answer", 42)
    method3_calls = [call("answer", 42), call("other", "xxx")]
    setitem3.assert_has_calls(method3_calls)


def test_execute_method_updates_monitor(
    mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
):
    loader = make_test_loader(mocker)
    method1 = make_test_method(mocker)
    mocker.patch.object(
        method1, "gpu_time", GpuTimeInfo(kernel=42.0, device2host=1.0, host2device=2.0)
    )
    mon = mocker.create_autospec(MonitoringInterface, instance=True)
    p = Pipeline(loader=loader, methods=[method1])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD, monitor=mon)
    t._prepare()
    mocker.patch.object(method1, "execute", return_value=dummy_block)
    t._execute_method(method1, dummy_block)

    mon.report_method_block.assert_called_once_with(
        method1.method_name,
        method1.module_path,
        method1.task_id,
        0,
        dummy_block.shape,
        dummy_block.chunk_index,
        dummy_block.global_index,
        ANY,
        42.0,
        2.0,
        1.0,
    )


def test_execute_section_calls_blockwise_execute_and_monitors(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: PathLike
):
    original_value = dummy_block.data[0, 0, 0]  # it has all the same number
    loader = make_test_loader(mocker, dummy_block)
    method = make_test_method(mocker, method_name="m1")
    p = Pipeline(loader=loader, methods=[method])
    s = sectionize(p)
    mon = mocker.create_autospec(MonitoringInterface, instance=True)
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD, monitor=mon)

    # Patch the store backing calculator function to assume being backed by RAM
    mocker.patch(
        "httomo.runner.task_runner.determine_store_backing",
        return_value=DataSetStoreBacking.RAM,
    )

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
    t._execute_section(s[0])
    assert isinstance(t.sink, DataSetStoreWriter)
    reader = t.sink.make_reader()
    data = reader.read_block(0, dummy_block.shape[0])

    np.testing.assert_allclose(data.data, original_value * 2)
    calls = [call(ANY, ANY), call(ANY, ANY)]
    block_mock.assert_has_calls(calls)  # make sure we got called twice
    assert mon.report_source_block.call_count == 2
    assert mon.report_sink_block.call_count == 2


def test_execute_section_for_block(
    mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
):
    loader = make_test_loader(mocker, dummy_block)
    method1 = make_test_method(mocker, method_name="m1")
    method2 = make_test_method(mocker, method_name="m2")
    p = Pipeline(loader=loader, methods=[method1, method2])
    s = sectionize(p)
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t._prepare()
    exec_method = mocker.patch.object(t, "_execute_method", return_value=dummy_block)
    t._execute_section_block(s[0], dummy_block)

    calls = [call(method1, ANY), call(method2, ANY)]
    exec_method.assert_has_calls(calls)


def test_does_reslice_when_needed_and_reports_time(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: PathLike
):
    loader = make_test_loader(mocker, dummy_block)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    mocker.patch.object(method1, "execute", return_value=dummy_block)
    method2 = make_test_method(mocker, method_name="m2", pattern=Pattern.sinogram)
    # create a block in the other slicing dim
    block2 = DataSetBlock(
        data=dummy_block.data,
        aux_data=dummy_block.aux_data,
        slicing_dim=1,
    )
    mocker.patch.object(method2, "execute", return_value=block2)
    p = Pipeline(loader=loader, methods=[method1, method2])
    mon = mocker.create_autospec(MonitoringInterface, instance=True)
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD, monitor=mon)

    # Patch the store backing calculator function to assume being backed by RAM
    mocker.patch(
        "httomo.runner.task_runner.determine_store_backing",
        return_value=DataSetStoreBacking.RAM,
    )
    t.execute()

    assert loader.pattern == Pattern.projection
    assert t.source is not None
    assert t.sink is not None
    assert t.source.slicing_dim == 1
    assert t.sink.slicing_dim == 1

    mon.report_total_time.assert_called_once()


def test_warns_with_multiple_reslices(
    mocker: MockerFixture,
    dummy_block: DataSetBlock,
    tmp_path: PathLike,
):
    loader = make_test_loader(mocker, dummy_block, pattern=Pattern.projection)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    method2 = make_test_method(mocker, method_name="m2", pattern=Pattern.sinogram)
    method3 = make_test_method(mocker, method_name="m3", pattern=Pattern.projection)
    method4 = make_test_method(mocker, method_name="m4", pattern=Pattern.sinogram)
    method5 = make_test_method(mocker, method_name="m5", pattern=Pattern.projection)
    p = Pipeline(loader=loader, methods=[method1, method2, method3, method4, method5])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)

    spy = mocker.patch("httomo.runner.task_runner.log_once")

    t._sectionize()

    spy.assert_called()
    args, _ = spy.call_args
    assert "Data saving or/and reslicing operation will be performed 4 times" in args[0]


def test_warns_with_multiple_stores_from_side_outputs(
    mocker: MockerFixture,
    dummy_block: DataSetBlock,
    tmp_path: PathLike,
):
    # Mock pipeline contains all projection methods, so no reslices occur. However, each method
    # requires side output from previous method, which causes data to be written to store after
    # each method
    loader = make_test_loader(mocker, dummy_block, pattern=Pattern.projection)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    method2 = make_test_method(
        mocker,
        method_name="m2",
        pattern=Pattern.projection,
        param1=OutputRef(method1, "method1_out"),
    )
    method3 = make_test_method(
        mocker,
        method_name="m3",
        pattern=Pattern.projection,
        param2=OutputRef(method2, "method2_out"),
    )
    method4 = make_test_method(
        mocker,
        method_name="m4",
        pattern=Pattern.projection,
        param3=OutputRef(method3, "method3_out"),
    )
    method5 = make_test_method(
        mocker,
        method_name="m5",
        pattern=Pattern.projection,
        param4=OutputRef(method4, "method4_out"),
    )

    p = Pipeline(loader=loader, methods=[method1, method2, method3, method4, method5])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)

    spy = mocker.patch("httomo.runner.task_runner.log_once")

    t._sectionize()

    spy.assert_called()
    args, _ = spy.call_args
    assert "Data saving or/and reslicing operation will be performed 4 times" in args[0]
