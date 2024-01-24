import logging
from os import PathLike
from typing import List
from unittest.mock import ANY, call
import pytest
import numpy as np
from pytest_mock import MockerFixture
import httomo
from httomo.data.dataset_store import DataSetStoreWriter
from httomo.logger import setup_logger
from httomo.runner.dataset import DataSet, DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.pipeline import Pipeline
from httomo.runner.platform_section import sectionize
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
        save_results_set=[False],
    )
    t = TaskRunner(p, reslice_dir=tmp_path)
    with pytest.raises(ValueError) as e:
        t._check_params_for_sweep()


def test_can_load_datasets(mocker: MockerFixture, tmp_path: PathLike):
    loader = make_test_loader(mocker)
    mksrc = mocker.patch.object(loader, "make_data_source")
    p = Pipeline(loader=loader, methods=[make_test_method(mocker)], save_results_set=[False])
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()

    mksrc.assert_called()
    assert t.source is not None


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_can_determine_max_slices_no_gpu_estimator(
    mocker: MockerFixture, tmp_path: PathLike, dummy_dataset: DataSet
):
    loader = make_test_loader(mocker, dummy_dataset)
    method = make_test_method(mocker, gpu=True)
    method.memory_gpu = []
    p = Pipeline(loader=loader, methods=[method], save_results_set=[False])
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(p, False)
    assert s[0].gpu is True

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == dummy_dataset.shape[0]


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
    dummy_dataset: DataSet,
):
    loader = make_test_loader(mocker, dummy_dataset)
    methods: List[MethodWrapper] = []
    calc_dims_mocks = []
    calc_max_slices_mocks = []
    save_results_list = []
    for i, max_slices in enumerate(max_slices_methods):
        method = make_test_method(mocker, gpu=True)
        method.memory_gpu = [
            GpuMemoryRequirement(dataset="tomo", multiplier=2.0, method="direct")
        ]
        calc_dims_mocks.append(
            mocker.patch.object(method, "calculate_output_dims", return_value=(10, 10))
        )
        calc_max_slices_mocks.append(
            mocker.patch.object(
                method, "calculate_max_slices", return_value=(max_slices, 1000000)
            )
        )
        methods.append(method)
        save_results_list.append(False)
    p = Pipeline(loader=loader, methods=methods, save_results_set=save_results_list)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(p, False)
    assert s[0].gpu is True
    assert len(s[0]) == len(max_slices_methods)
    shape = (dummy_dataset.shape[1], dummy_dataset.shape[2])

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == min(max_slices_methods)
    for i in range(len(max_slices_methods)):
        calc_dims_mocks[i].assert_called_with(shape)
        calc_max_slices_mocks[i].assert_called_with(
            dummy_dataset.data.dtype,
            shape,
            ANY,
            dummy_dataset.darks,
            dummy_dataset.flats,
        )


def test_can_determine_max_slices_with_cpu(
    mocker: MockerFixture, tmp_path: PathLike, dummy_dataset: DataSet
):
    loader = make_test_loader(mocker, dummy_dataset)
    methods = []
    for _ in range(3):
        method = make_test_method(mocker, gpu=False)
        methods.append(method)
    p = Pipeline(loader=loader, methods=methods, save_results_set=[False, False, False])
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    s = sectionize(p, False)
    assert s[0].gpu is False

    t.determine_max_slices(s[0], 0)
    assert s[0].max_slices == dummy_dataset.shape[0]


def test_calls_update_side_inputs_after_call(
    mocker: MockerFixture, tmp_path: PathLike, dummy_dataset: DataSet
):
    loader = make_test_loader(mocker, dummy_dataset)

    side_outputs = {"answer": 42, "other": "xxx"}

    block = dummy_dataset.make_block(0)
    method1 = make_test_method(mocker)
    mocker.patch.object(method1, "execute", return_value=block)
    mocker.patch.object(method1, "get_side_output", return_value=side_outputs)
    method2 = make_test_method(mocker)

    p = Pipeline(loader=loader, methods=[method1, method2], save_results_set=[False, False])
    t = TaskRunner(p, reslice_dir=tmp_path)
    spy = mocker.patch.object(t, "update_side_inputs")
    t._prepare()
    t._execute_method(method1, 2, block)

    spy.assert_called_with(side_outputs)
    t.side_outputs == side_outputs


def test_update_side_inputs_updates_downstream_methods(
    mocker: MockerFixture, tmp_path: PathLike
):
    loader = make_test_loader(mocker)
    side_outputs = {"answer": 42, "other": "xxx"}
    method1 = make_test_method(mocker, method_name="m1")
    method2 = make_test_method(mocker, method_name="m2")
    method2.parameters = ["answer"]
    setitem2 = mocker.patch.object(method2, "__setitem__")
    method3 = make_test_method(mocker, method_name="m3")
    method3.parameters = ["answer", "other", "whatever"]
    setitem3 = mocker.patch.object(method3, "__setitem__")

    p = Pipeline(
        loader=loader,
        methods=[method1, method2, method3],
        save_results_set=[False, False, False],
    )
    t = TaskRunner(p, reslice_dir=tmp_path)
    t.method_index = 2  # pretend we're after executing method1
    t.update_side_inputs(side_outputs)

    setitem2.assert_called_with("answer", 42)
    method3_calls = [call("answer", 42), call("other", "xxx")]
    setitem3.assert_has_calls(method3_calls)


def test_execute_section_calls_blockwise_execute(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike
):
    original_value = dummy_dataset.data[0, 0, 0]  # it has all the same number
    loader = make_test_loader(mocker, dummy_dataset)
    method = make_test_method(mocker, method_name="m1")
    p = Pipeline(loader=loader, methods=[method], save_results_set=[False])
    s = sectionize(p, False)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    # make that do nothing
    mocker.patch.object(t, "determine_max_slices")
    s[0].max_slices = dummy_dataset.data.shape[0] / 2  # we'll have 2 blocks

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
    data = reader.read_block(0, dummy_dataset.shape[0])

    np.testing.assert_allclose(data.data, original_value * 2)
    calls = [call(ANY, ANY), call(ANY, ANY)]
    block_mock.assert_has_calls(calls)  # make sure we got called twice


def test_execute_section_for_block(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike
):
    loader = make_test_loader(mocker, dummy_dataset)
    method1 = make_test_method(mocker, method_name="m1")
    method2 = make_test_method(mocker, method_name="m2")
    p = Pipeline(
        loader=loader, methods=[method1, method2], save_results_set=[False, False]
    )
    s = sectionize(p, False)
    t = TaskRunner(p, reslice_dir=tmp_path)
    t._prepare()
    block = dummy_dataset.make_block(0)
    exec_method = mocker.patch.object(t, "_execute_method", return_value=block)
    t._execute_section_block(s[0], block)

    calls = [call(method1, ANY, ANY), call(method2, ANY, ANY)]
    exec_method.assert_has_calls(calls)


def test_does_reslice_when_needed(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike
):
    loader = make_test_loader(mocker, dummy_dataset)
    block = dummy_dataset.make_block(0)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    mocker.patch.object(method1, "execute", return_value=block)
    method2 = make_test_method(mocker, method_name="m2", pattern=Pattern.sinogram)
    block2 = dummy_dataset.make_block(1)
    mocker.patch.object(method2, "execute", return_value=block2)
    p = Pipeline(loader=loader, methods=[method1, method2], save_results_set=[False, False])
    t = TaskRunner(p, reslice_dir=tmp_path)

    t.execute()

    assert loader.pattern == Pattern.projection
    assert t.source is not None
    assert t.sink is not None
    assert t.source.slicing_dim == 1
    assert t.sink.slicing_dim == 1


@pytest.mark.parametrize("loader_pattern,reslices", [
    (Pattern.all, 2),
    (Pattern.projection, 2),
    (Pattern.sinogram, 3)
])
def test_warns_with_multiple_reslices(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike, loader_pattern: Pattern,
    reslices: int
):
    loader = make_test_loader(mocker, dummy_dataset, pattern=loader_pattern)
    method1 = make_test_method(mocker, method_name="m1", pattern=Pattern.projection)
    method2 = make_test_method(mocker, method_name="m2", pattern=Pattern.sinogram)
    method3 = make_test_method(mocker, method_name="m3", pattern=Pattern.projection)
    p = Pipeline(
        loader=loader,
        methods=[method1, method2, method3],
        save_results_set=[False, False, False],
    )
    t = TaskRunner(p, reslice_dir=tmp_path)

    spy = mocker.patch("httomo.runner.task_runner.log_once")

    t._sectionize()

    spy.assert_called()
    args, kwargs = spy.call_args
    assert f"Reslicing will be performed {reslices} times" in args[0]


@pytest.mark.parametrize(
    "method_name,dim,save",
    [
        ("m1", 3, True),
        ("m1", 2, False),
        ("save_to_images", 3, False),
        ("find_center_vo", 3, False),
    ],
)
def test_saves_intermediate_results(
    mocker: MockerFixture,
    dummy_dataset: DataSet,
    method_name: str,
    dim: int,
    save: bool,
):
    if dim != 3:
        dummy_dataset.data = np.squeeze(dummy_dataset.data[0, :, :])
    loader = make_test_loader(mocker)
    loader.load.return_value = dummy_dataset
    loader.detector_x = 15
    loader.detector_y = 42
    method1 = make_test_method(
        mocker,
        method_name=method_name,
        module_path="path1",
    )
    method1.recon_algorithm = "testalgo"
    method2 = make_test_method(mocker, method_name="m2")
    p = Pipeline(
        loader=loader, methods=[method1, method2], save_results_set=[True, False]
    )
    t = TaskRunner(p)

    exec_section = mocker.patch.object(t, "_execute_section")
    intermediate_save = mocker.patch("httomo.runner.task_runner.intermediate_dataset")

    t.execute()

    exec_section.assert_has_calls([call(ANY, 0), call(ANY, 1)])
    if save:
        intermediate_save.assert_called_once_with(
            ANY,  # data
            ANY,  # run_out_dir
            ANY,  # angles
            15,  # detector_x
            42,  # detector_y
            ANY,  # comm
            2,  # method index: loader + method1
            method1.package_name,  # package_name
            method1.method_name,  # method_name
            "tomo",  # dataset name
            1,  # slicing dim
            "testalgo",  # algo name
        )
    else:
        intermediate_save.assert_not_called()
