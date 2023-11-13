from typing import List
from unittest.mock import ANY, call
import pytest
import numpy as np
from pytest_mock import MockerFixture
from httomo.runner.dataset import DataSet
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.pipeline import Pipeline
from httomo.runner.platform_section import sectionize
from httomo.runner.task_runner import TaskRunner
from httomo.utils import xp, gpu_enabled
from .testing_utils import make_test_loader, make_test_method


def test_check_params_for_sweep_raises_exception(mocker: MockerFixture):
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
    t = TaskRunner(p)
    with pytest.raises(ValueError) as e:
        t._check_params_for_sweep()


def test_can_load_datasets(mocker: MockerFixture):
    loader = make_test_loader(mocker)
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        darks=np.ones((10, 10)),
        flats=np.ones((10, 10)),
        angles=np.ones((10,)),
    )
    loader.load.return_value = dataset
    p = Pipeline(loader=loader, methods=[make_test_method(mocker)])
    t = TaskRunner(p)
    t._prepare()

    loader.load.assert_called()
    assert t.dataset is not None
    np.testing.assert_array_equal(t.dataset.data, 1)


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_can_determine_max_slices_no_gpu_estimator(mocker: MockerFixture):
    loader = make_test_loader(mocker)
    dataset = DataSet(
        data=np.ones((10, 10, 10)),
        darks=np.ones((10, 10)),
        flats=np.ones((10, 10)),
        angles=np.ones((10,)),
    )
    loader.load.return_value = dataset
    method = make_test_method(mocker, gpu=True)
    method.memory_gpu = []
    p = Pipeline(loader=loader, methods=[method])
    t = TaskRunner(p)
    t._prepare()
    s = sectionize(p, False)
    assert s[0].gpu is True

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == dataset.data.shape[0]


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
    mocker: MockerFixture, dummy_dataset: DataSet, max_slices_methods: List[int]
):
    loader = make_test_loader(mocker)
    loader.load.return_value = dummy_dataset
    methods = []
    for i, max_slices in enumerate(max_slices_methods):
        method = make_test_method(mocker, gpu=True)
        method.memory_gpu = [
            GpuMemoryRequirement(dataset="tomo", multiplier=2.0, method="direct")
        ]
        method.calculate_output_dims.return_value = (10, 10)
        method.calculate_max_slices.return_value = (max_slices, 1000000)
        methods.append(method)
    p = Pipeline(loader=loader, methods=methods)
    t = TaskRunner(p)
    t._prepare()
    s = sectionize(p, False)
    assert s[0].gpu is True
    assert len(s[0]) == len(max_slices_methods)
    shape = list(dummy_dataset.data.shape)
    shape.pop(0)

    t.determine_max_slices(s[0], 0)

    assert s[0].max_slices == min(max_slices_methods)
    methods[0].calculate_output_dims.assert_called_with(tuple(shape))
    methods[0].calculate_max_slices.assert_called_with(dummy_dataset, tuple(shape), ANY)
    for i in range(1, len(max_slices_methods)):
        methods[i].calculate_output_dims.assert_called_with((10, 10))
        methods[i].calculate_max_slices.assert_called_with(dummy_dataset, (10, 10), ANY)


def test_calls_update_side_inputs_after_call(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    loader = make_test_loader(mocker)
    loader.load.return_value = dummy_dataset

    side_outputs = {"answer": 42, "other": "xxx"}

    method1 = make_test_method(mocker)
    method1.execute.return_value = dummy_dataset
    method1.get_side_output.return_value = side_outputs
    method2 = make_test_method(mocker)

    p = Pipeline(loader=loader, methods=[method1, method2])
    t = TaskRunner(p)
    spy = mocker.patch.object(t, "update_side_inputs")
    t._prepare()
    t._execute_method(method1, 2, dummy_dataset)

    spy.assert_called_with(side_outputs)
    t.side_outputs == side_outputs


def test_update_side_inputs_updates_downstream_methods(mocker: MockerFixture):
    loader = make_test_loader(mocker)
    side_outputs = {"answer": 42, "other": "xxx"}
    method1 = make_test_method(mocker, method_name="m1")
    method2 = make_test_method(mocker, method_name="m2")
    method2.parameters = ["answer"]
    method3 = make_test_method(mocker, method_name="m3")
    method3.parameters = ["answer", "other", "whatever"]

    p = Pipeline(loader=loader, methods=[method1, method2, method3])
    t = TaskRunner(p)
    t.method_index = 2  # pretend we're after executing method1
    t.update_side_inputs(side_outputs)

    method2.__setitem__.assert_called_with("answer", 42)
    method3_calls = [call("answer", 42), call("other", "xxx")]
    method3.__setitem__.assert_has_calls(method3_calls)


def test_execute_section_calls_blockwise_execute(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    original_value = dummy_dataset.data[0, 0, 0]  # it has all the same number
    loader = make_test_loader(mocker)
    loader.load.return_value = dummy_dataset
    method = make_test_method(mocker, method_name="m1")
    p = Pipeline(loader=loader, methods=[method])
    s = sectionize(p, False)
    t = TaskRunner(p)
    t._prepare()
    # make that do nothing
    mocker.patch.object(t, "determine_max_slices")
    s[0].max_slices = dummy_dataset.data.shape[0] / 2  # we'll have 2 blocks

    # make this function return the block, with data multiplied by 2
    def mul_block_by_two(section, block: DataSet):
        block.data *= 2
        return block

    block_mock = mocker.patch.object(
        t, "_execute_section_block", side_effect=mul_block_by_two
    )

    t._execute_section(s[0], 1)

    np.testing.assert_allclose(t.dataset.data, original_value * 2)
    calls = [call(ANY, ANY), call(ANY, ANY)]
    block_mock.assert_has_calls(calls)  # make sure we got called twice
