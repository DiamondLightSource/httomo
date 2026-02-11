from os import PathLike
from pathlib import Path
from typing import List, Tuple
from unittest.mock import ANY, call

import h5py
import pytest
import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture

import httomo.globals
from httomo.darks_flats import DarksFlatsFileConfig
from httomo.data.dataset_store import DataSetStoreWriter
from httomo.loaders import make_loader
from httomo.loaders.types import RawAngles
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.save_intermediate import SaveIntermediateFilesWrapper
from httomo.preview import PreviewConfig, PreviewDimConfig
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_backing import DataSetStoreBacking
from httomo.runner.methods_repository_interface import MethodQuery
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
from ..testing_utils import make_mock_preview_config, make_test_loader, make_test_method

from httomo_backends.methods_database.query import GpuMemoryRequirement


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
    loader = make_test_loader(mocker, block=dummy_block)
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
    loader = make_test_loader(mocker, block=dummy_block)
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
    loader = make_test_loader(mocker, block=dummy_block)
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
    loader = make_test_loader(mocker, block=dummy_block)
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
    loader = make_test_loader(mocker, block=block)
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


def test_determine_max_slices_raises_error_if_padded_data_cannot_fit(
    mocker: MockerFixture, tmp_path: PathLike
):
    METHOD_NAME = "method_requiring_padding"
    PADDING = (5, 5)
    MAX_SLICES_LESS_THAN_CORE_PLUS_PADDING = 9
    data = np.ones((10, 10, 10), dtype=np.float32)
    aux = AuxiliaryData(angles=np.ones(data.shape[0], dtype=np.float32))
    block = DataSetBlock(data, aux)
    loader = make_test_loader(mocker, block=block, padding=PADDING)
    padded_gpu_method = make_test_method(
        mocker,
        method_name=METHOD_NAME,
        gpu=True,
        padding=True,
        memory_gpu=GpuMemoryRequirement(multiplier=2.0, method="direct"),
    )
    mocker.patch.object(
        padded_gpu_method,
        "calculate_max_slices",
        return_value=(MAX_SLICES_LESS_THAN_CORE_PLUS_PADDING, 0),
    )
    mocker.patch.object(
        padded_gpu_method,
        "calculate_output_dims",
        return_value=(data.shape[1], data.shape[2]),
    )

    p = Pipeline(loader=loader, methods=[padded_gpu_method])
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    t._prepare()
    s = sectionize(p)

    err_str = (
        "Unable to process data due to GPU memory limitations.\n"
        f"Please remove method '{METHOD_NAME}' from the pipeline, or run on a machine with "
        "more GPU memory."
    )
    with pytest.raises(ValueError, match=err_str):
        t.determine_max_slices(s[0], 0)


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
        method,
        block1,
    )  # the first block shouldn't trigger a side output append call
    assert spy.call_count == 0

    t._execute_method(
        method,
        block2,
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


def test_execute_section_block_updates_monitor(
    mocker: MockerFixture, tmp_path: PathLike, dummy_block: DataSetBlock
):
    loader = make_test_loader(mocker)
    method1 = make_test_method(mocker)
    mocker.patch.object(
        method1, "gpu_time", GpuTimeInfo(kernel=42.0, device2host=1.0, host2device=2.0)
    )
    mon = mocker.create_autospec(MonitoringInterface, instance=True)
    p = Pipeline(loader=loader, methods=[method1])
    s = sectionize(p)
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD, monitor=mon)
    t._prepare()
    mocker.patch.object(method1, "execute", return_value=dummy_block)
    t._execute_section_block(s[0], dummy_block)

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
    loader = make_test_loader(mocker, block=dummy_block)
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
    loader = make_test_loader(mocker, block=dummy_block)
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
    loader = make_test_loader(mocker, block=dummy_block)
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
    loader = make_test_loader(mocker, block=dummy_block, pattern=Pattern.projection)
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


def test_get_method_names_for_snapshot_saver(
    mocker: MockerFixture,
    dummy_block: DataSetBlock,
    tmp_path: PathLike,
):
    loader = make_test_loader(mocker, block=dummy_block, pattern=Pattern.projection)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    method2 = make_test_method(mocker, method_name="m2_rec", pattern=Pattern.projection)
    method3 = make_test_method(
        mocker, method_name="data_checker", pattern=Pattern.projection
    )
    method4 = make_test_method(
        mocker, method_name="find_center_pc", pattern=Pattern.sinogram
    )
    method5 = make_test_method(mocker, method_name="m5_rec", pattern=Pattern.sinogram)
    method6 = make_test_method(
        mocker, method_name="data_checker", pattern=Pattern.sinogram
    )
    method7 = make_test_method(mocker, method_name="m7_rec", pattern=Pattern.projection)
    method8 = make_test_method(
        mocker, method_name="data_checker", pattern=Pattern.projection
    )
    method9 = make_test_method(mocker, method_name="m9_rec", pattern=Pattern.sinogram)
    method10 = make_test_method(
        mocker, method_name="data_checker", pattern=Pattern.sinogram
    )
    method11 = make_test_method(
        mocker, method_name="calculate_stats", pattern=Pattern.all
    )
    p = Pipeline(
        loader=loader,
        methods=[
            method1,
            method2,
            method3,
            method4,
            method5,
            method6,
            method7,
            method8,
            method9,
            method10,
            method11,
        ],
    )
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)
    _sections = t._sectionize()

    sections_number = len(_sections)
    METHODS_NAMES_EXPECTED = ["m2_rec", "m5_rec", "m7_rec", "m9_rec"]
    for ind in range(0, sections_number):
        assert (
            t._get_methods_name_for_snapshot(_sections[ind])
            == METHODS_NAMES_EXPECTED[ind]
        )


def test_warns_with_multiple_stores_from_side_outputs(
    mocker: MockerFixture,
    dummy_block: DataSetBlock,
    tmp_path: PathLike,
):
    # Mock pipeline contains all projection methods, so no reslices occur. However, each method
    # requires side output from previous method, which causes data to be written to store after
    # each method
    loader = make_test_loader(mocker, block=dummy_block, pattern=Pattern.projection)
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


def test_execute_section_with_padding_produces_correct_result(
    mocker: MockerFixture,
    standard_data_path: str,
    standard_image_key_path: str,
    tmp_path: PathLike,
):
    # Define loader wrapper to load standard test data
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    DARKS_FLATS_CONFIG = DarksFlatsFileConfig(
        file=IN_FILE_PATH,
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        ignore=False,
    )
    ANGLES_CONFIG = RawAngles(data_path="/entry1/tomo_entry/data/rotation_angle")
    COMM = MPI.COMM_WORLD
    DATA_SHAPE = (180, 128, 160)
    PREVIEW_CONFIG = PreviewConfig(
        # Use only a small number of projections, for the sake of making the test not take too
        # long to run (the dummy 3D method below is inefficiently implemented)
        angles=PreviewDimConfig(start=0, stop=10),
        detector_y=PreviewDimConfig(start=0, stop=DATA_SHAPE[1]),
        detector_x=PreviewDimConfig(start=0, stop=DATA_SHAPE[2]),
    )
    mock_repo = mocker.MagicMock()
    loader_wrapper = make_loader(
        repo=mock_repo,
        module_path="httomo.data.hdf.loaders",
        method_name="standard_tomo",
        in_file=IN_FILE_PATH,
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        angles=ANGLES_CONFIG,
        comm=COMM,
        darks=DARKS_FLATS_CONFIG,
        flats=DARKS_FLATS_CONFIG,
        preview=PREVIEW_CONFIG,
    )

    # Define dummy method function that depends on padded slices/neighbourhood
    class FakeMethodsModule:
        def method_using_padding_slices(data: np.ndarray, kernel_size: int):
            """
            This is intended to be a very simple 3D method that depends on the padding
            slices/neighbourhood, used to detect if padding is being applied correctly by the
            runner in the test.

            Note that the implementation is not intended to be efficient, but instead to be
            clear that it's dependent on padding slices providing the required neighbourhood.
            Therefore, the implementation's performance is certainly rather poor!
            """
            out = np.empty_like(data)
            # For each pixel, get 3D neighbourhood around it and perform a sum over the pixels
            # in the neighbourhood. The size of the neighbourhood is defined by the
            # `kernel_size` parameter.
            radius = kernel_size // 2
            for i in range(data.shape[0]):
                angles_start = max(0, i - radius)
                angles_stop = min(data.shape[0], i + radius)
                for j in range(data.shape[1]):
                    det_y_start = max(0, j - radius)
                    det_y_stop = min(data.shape[1], j + radius)
                    for k in range(data.shape[2]):
                        det_x_start = max(0, k - radius)
                        det_x_stop = min(data.shape[2], k + radius)
                        neighbourhood = data[
                            angles_start:angles_stop,
                            det_y_start:det_y_stop,
                            det_x_start:det_x_stop,
                        ]
                        out[i, j, k] = neighbourhood.sum()
            return out

    # Create method wrapper
    KERNEL_SIZE = 3
    MODULE_PATH = "module_path"
    METHOD_NAME = "method_using_padding_slices"
    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeMethodsModule
    )
    method_query = mocker.create_autospec(MethodQuery)
    mocker.patch.object(target=method_query, attribute="padding", return_value=True)
    mocker.patch.object(
        target=method_query,
        attribute="calculate_padding",
        return_value=(KERNEL_SIZE // 2, KERNEL_SIZE // 2),
    )
    mocker.patch.object(
        target=method_query, attribute="get_pattern", return_value=Pattern.projection
    )
    mocker.patch.object(
        target=method_query, attribute="get_output_dims_change", return_value=False
    )
    mocker.patch.object(
        target=method_query, attribute="get_implementation", return_value="cpu"
    )
    mocker.patch.object(
        target=method_query, attribute="get_memory_gpu_params", return_value=None
    )
    mocker.patch.object(
        target=method_query, attribute="save_result_default", return_value=False
    )
    mocker.patch.object(
        target=method_query, attribute="swap_dims_on_output", return_value=False
    )
    mocker.patch.object(target=mock_repo, attribute="query", return_value=method_query)
    wrapper = make_method_wrapper(
        method_repository=mock_repo,
        module_path=MODULE_PATH,
        method_name=METHOD_NAME,
        comm=COMM,
        preview_config=make_mock_preview_config(mocker),
        kernel_size=KERNEL_SIZE,
    )

    # Define a pipeline containing one padding method
    pipeline = Pipeline(loader=loader_wrapper, methods=[wrapper])

    # Create task runner object + prepare
    runner = TaskRunner(pipeline=pipeline, reslice_dir=tmp_path, comm=COMM)
    runner._prepare()

    # Patch `determine_max_slices()` to do nothing (so then the max slices can be set
    # on the first section manually later)
    mocker.patch.object(target=runner, attribute="determine_max_slices")

    sections = sectionize(pipeline)

    # Force the chunk to be split into multiple blocks for processing (so then the presence of
    # padding slices will be detectable if padding is incorrectly applied by the runner)
    MAX_SLICES = PREVIEW_CONFIG.angles.stop // 2
    sections[0].max_slices = MAX_SLICES

    # Set `is_last=False` on section object to force writing to a `DataSetStoreWriter` instead
    # of a `DummySink`
    #
    # NOTE: This makes an assumption about the internals of the task runner, which isn't great.
    # See note below when asserting that `runner.sink` is an instance of `DataSetStoreWriter`
    # (in order to inspect the output data) for more info.
    sections[0].is_last = False

    # Execute the single section that contains the wrapper with the dummy 3D method
    runner._execute_section(sections[0])

    # Get subset of projection data (running on a subset because to check that padding is
    # applied correctly doesn't require lots of data, and because the dummy 3D method
    # implementation is very inefficient so processing lots of data will make the test slow)
    with h5py.File(IN_FILE_PATH, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        projections = dataset[
            PREVIEW_CONFIG.angles.start : PREVIEW_CONFIG.angles.stop,
            PREVIEW_CONFIG.detector_y.start : PREVIEW_CONFIG.detector_y.stop,
            PREVIEW_CONFIG.detector_x.start : PREVIEW_CONFIG.detector_x.stop,
        ]

    # Pad the projection data by however much is needed by the dummy 3D method
    padding = wrapper.calculate_padding()
    projections = np.pad(
        projections,
        pad_width=(padding, (0, 0), (0, 0)),
        mode="edge",
    )

    # Execute the dummy method with the full projection data
    expected_output = FakeMethodsModule.method_using_padding_slices(
        data=projections,
        kernel_size=KERNEL_SIZE,
    )

    # NOTE: Assuming the `runner.sink` attribute is `DataSetStoreWriter`, in conjunction with
    # setting `section.is_last = False` earlier is assuming that the `DataSetSink` implementor
    # for `runner.sink` is `DataSetStoreWriter` for any section which has `section.is_last =
    # False`.
    #
    # It's not good to assume the implementor of `DataSetSink` that is stored in
    # `runner.sink`, but currently it seems that there's no way to check the data without
    # digging into the internals at the moment.
    assert isinstance(runner.sink, DataSetStoreWriter)
    reader = runner.sink.make_reader()
    runner_result = reader.read_block(
        start=0, length=PREVIEW_CONFIG.angles.stop - PREVIEW_CONFIG.angles.start
    )

    # Compare contents of reader (result of processing multiple padded blocks that form the
    # chunk) to expected result (result of processing the entire chunk in one go)
    np.testing.assert_array_equal(
        runner_result.data_unpadded,
        expected_output[padding[0] : PREVIEW_CONFIG.angles.stop + padding[1], :, :],
    )


def test_passes_minimum_block_length_to_intermediate_data_wrapper(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: PathLike
):
    loader = make_test_loader(mocker, block=dummy_block)
    method = make_test_method(mocker, method_name="m1", save_result=True)
    method_spy = mocker.patch.object(method, "append_config_params")
    mocker.patch.object(httomo.globals, "run_out_dir", tmp_path)
    saver = mocker.create_autospec(
        SaveIntermediateFilesWrapper,
        instance=True,
        gpu=False,
        pattern=Pattern.projection,
        module_path="httomo.methods",
        method_name="save_intermediate_data",
        memory_gpu=None,
        save_result=False,
        task_id=None,
        padding=False,
        sweep=False,
    )
    saver_spy = mocker.patch.object(saver, "append_config_params")
    p = Pipeline(loader=loader, methods=[method, saver])
    s = sectionize(p)
    t = TaskRunner(p, reslice_dir=tmp_path, comm=MPI.COMM_WORLD)

    # Patch the store backing calculator function to assume being backed by RAM
    mocker.patch(
        "httomo.runner.task_runner.determine_store_backing",
        return_value=DataSetStoreBacking.RAM,
    )

    t._prepare()

    # Avoid max slices calculation and manually choose it instead
    mocker.patch.object(t, "determine_max_slices")
    s[0].max_slices = dummy_block.chunk_shape[0]

    # Execute section, check that only the intermediate data wrapper in the section was given
    # the minimum block length value as a parameter
    t._execute_section(s[0])
    method_spy.assert_not_called()
    saver_spy.assert_called_once_with(
        {"minimum_block_length": dummy_block.chunk_shape[0]}
    )
