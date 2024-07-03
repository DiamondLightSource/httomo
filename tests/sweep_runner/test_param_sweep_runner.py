import pytest
from pytest_mock import MockerFixture

import numpy as np
from mpi4py import MPI

from httomo.method_wrappers import make_method_wrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.pipeline import Pipeline
from httomo.sweep_runner.param_sweep_runner import ParamSweepRunner
from httomo.sweep_runner.side_output_manager import SideOutputManager
from httomo.sweep_runner.stages import Stages
from tests.testing_utils import make_mock_repo, make_test_loader, make_test_method


def test_without_prepare_block_property_raises_error(mocker: MockerFixture):
    loader = make_test_loader(mocker)
    pipeline = Pipeline(loader=loader, methods=[])
    stages = Stages(before_sweep=[], sweep=[], after_sweep=[])
    runner = ParamSweepRunner(pipeline=pipeline, stages=stages)
    with pytest.raises(ValueError) as e:
        runner.block
    assert "Block from input data has not yet been loaded" in str(e)


def test_after_prepare_block_attr_contains_data(mocker: MockerFixture):
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.arange(np.prod(PREVIEWED_SLICES_SHAPE), dtype=np.uint16).reshape(
        PREVIEWED_SLICES_SHAPE
    )
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)
    pipeline = Pipeline(loader=loader, methods=[])
    stages = Stages(before_sweep=[], sweep=[], after_sweep=[])
    runner = ParamSweepRunner(pipeline=pipeline, stages=stages)
    runner.prepare()
    assert runner.block is not None
    np.testing.assert_array_equal(runner.block.data, data)


def tests_prepare_raises_error_if_too_many_sino_slices(mocker: MockerFixture):
    TOO_MANY_SINO_SLICES = 10
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, TOO_MANY_SINO_SLICES, 160)
    data = np.arange(np.prod(PREVIEWED_SLICES_SHAPE), dtype=np.uint16).reshape(
        PREVIEWED_SLICES_SHAPE
    )
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)
    pipeline = Pipeline(loader=loader, methods=[])
    stages = Stages(before_sweep=[], sweep=[], after_sweep=[])
    runner = ParamSweepRunner(pipeline=pipeline, stages=stages)

    with pytest.raises(ValueError) as e:
        runner.prepare()

    err_str = (
        "Parameter sweep runs support input data containing <= 7 sinogram slices, "
        "input data contains 10 slices"
    )
    assert err_str in str(e)


@pytest.mark.parametrize("non_sweep_stage", ["before", "after"])
def test_execute_non_sweep_stage_modifies_block(
    mocker: MockerFixture,
    non_sweep_stage: str,
):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.ones(PREVIEWED_SLICES_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)

    # Define two dummy methods to be in the "before sweep" stage in the runner
    def dummy_method_1(block):
        block.data = block.data * 2
        return block

    def dummy_method_2(block):
        block.data = block.data * 3
        return block

    # Patch `the `execute() method in the mock method wrapper objects so then the functions
    # that are used for the two methods executed in the stage are the two dummy methods defined
    # above
    m1 = make_test_method(mocker=mocker, method_name="method_1")
    mocker.patch.object(target=m1, attribute="execute", side_effect=dummy_method_1)
    m2 = make_test_method(mocker=mocker, method_name="method_2")
    mocker.patch.object(target=m2, attribute="execute", side_effect=dummy_method_2)

    # Define pipeline, stages, runner objects, and execute the methods in the stage
    # before/after the sweep
    #
    # NOTE: the pipeline object is only needed for providing the loader, and the methods needed
    # for the purpose of this test are only the ones in the "before/after sweep" stage, which
    # is why the pipeline and stages object both have only partial pipeline information
    pipeline = Pipeline(loader=loader, methods=[])
    if non_sweep_stage == "before":
        stages = Stages(
            before_sweep=[m1, m2],
            sweep=[],
            after_sweep=[],
        )
    else:
        stages = Stages(
            before_sweep=[],
            sweep=[],
            after_sweep=[m1, m2],
        )

    runner = ParamSweepRunner(pipeline=pipeline, stages=stages)
    runner.prepare()
    if non_sweep_stage == "before":
        runner.execute_before_sweep()
    else:
        runner.execute_after_sweep()

    # Inspect the block data after the "before sweep" stage has completed, asserting that the
    # data reflects what the combination of two dummy methods in that stage should produce
    assert runner.block.data.shape == PREVIEWED_SLICES_SHAPE
    expected_block_data = data * 2 * 3
    np.testing.assert_array_equal(runner.block.data, expected_block_data)


@pytest.mark.parametrize("non_sweep_stage", ["before", "after"])
def test_execute_non_sweep_stage_method_output_updates_side_outputs(
    mocker: MockerFixture,
    non_sweep_stage: str,
):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.ones(PREVIEWED_SLICES_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)

    # Define side output that dummy method in stage will produce
    SIDE_OUTPUT_LABEL = "some_side_output"
    side_outputs = {SIDE_OUTPUT_LABEL: 5}

    # Define mock method wrapper to produce the above side output
    mock_wrapper = make_test_method(mocker)
    mocker.patch.object(
        target=mock_wrapper, attribute="get_side_output", return_value=side_outputs
    )

    # Define pipeline, stages, runner objects, and execute the methods in the stage
    # before/after the sweep
    #
    # NOTE: the pipeline object is only needed for providing the loader, and the methods needed
    # for the purpose of this test are only the ones in the "before/after sweep" stage, which
    # is why the pipeline and stages object both have only partial pipeline information
    pipeline = Pipeline(loader=loader, methods=[])
    if non_sweep_stage == "before":
        stages = Stages(
            before_sweep=[mock_wrapper],
            sweep=[],
            after_sweep=[],
        )
    else:
        stages = Stages(
            before_sweep=[],
            sweep=[],
            after_sweep=[mock_wrapper],
        )

    # Pass in an empty side output manager object into the runner. The execution of the method
    # should add the side output produced by it to the side output manager.
    side_output_manager = SideOutputManager()
    runner = ParamSweepRunner(
        pipeline=pipeline, stages=stages, side_output_manager=side_output_manager
    )
    runner.prepare()
    if non_sweep_stage == "before":
        runner.execute_before_sweep()
    else:
        runner.execute_after_sweep()

    # Check that the side output manager has been updated with the mock method's side output
    assert SIDE_OUTPUT_LABEL in side_output_manager.labels
    assert side_output_manager.get(SIDE_OUTPUT_LABEL) == side_outputs[SIDE_OUTPUT_LABEL]


@pytest.mark.parametrize("non_sweep_stage", ["before", "after"])
def test_execute_non_sweep_stage_method_params_updated_from_side_outputs(
    mocker: MockerFixture,
    non_sweep_stage: str,
):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.ones(PREVIEWED_SLICES_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)

    # Define side output that dummy method in stage will require
    SIDE_OUTPUT_LABEL = "some_side_output"
    side_outputs = {SIDE_OUTPUT_LABEL: 5}
    side_output_manager = SideOutputManager()
    side_output_manager.append(side_outputs)

    # Define mock module + mock method function that will be imported by method wrapper.
    #
    # The mock method function asserts that the param value it's given is the value of the side
    # output that it requires.
    class FakeModule:
        def method_1(data, some_side_output) -> np.ndarray:  # type: ignore
            assert some_side_output == side_outputs[SIDE_OUTPUT_LABEL]
            return np.empty(PREVIEWED_SLICES_SHAPE, dtype=np.float32)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="",
        method_name="method_1",
        comm=MPI.COMM_WORLD,
    )

    # Define pipeline, stages, runner objects, and execute the methods in the stage
    # before/after the sweep
    #
    # NOTE: the pipeline object is only needed for providing the loader, and the methods needed
    # for the purpose of this test are only the ones in the "before/after sweep" stage, which
    # is why the pipeline and stages object both have only partial pipeline information
    pipeline = Pipeline(loader=loader, methods=[])
    if non_sweep_stage == "before":
        stages = Stages(
            before_sweep=[method_wrapper],
            sweep=[],
            after_sweep=[],
        )
    else:
        stages = Stages(
            before_sweep=[],
            sweep=[],
            after_sweep=[method_wrapper],
        )

    runner = ParamSweepRunner(
        pipeline=pipeline, stages=stages, side_output_manager=side_output_manager
    )
    runner.prepare()
    if non_sweep_stage == "before":
        runner.execute_before_sweep()
    else:
        runner.execute_after_sweep()


def test_execute_sweep_stage_modifies_block(mocker: MockerFixture):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.ones(PREVIEWED_SLICES_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)

    # Define four dummy functions that are representing a single method configured with
    # different parameter values
    def method_sweep_val_1(block):
        block.data = block.data * 2
        return block

    def method_sweep_val_2(block):
        block.data = block.data * 3
        return block

    def method_sweep_val_3(block):
        block.data = block.data * 5
        return block

    def method_sweep_val_4(block):
        block.data = block.data * 7
        return block

    # Patch `the `execute() method in the mock method wrapper objects so then the functions
    # that are used for the three methods executed in the sweep stage are the three dummy
    # methods defined above
    m1 = make_test_method(mocker=mocker, method_name="method_sweep_val_1")
    mocker.patch.object(target=m1, attribute="execute", side_effect=method_sweep_val_1)
    m2 = make_test_method(mocker=mocker, method_name="method_sweep_val_2")
    mocker.patch.object(target=m2, attribute="execute", side_effect=method_sweep_val_2)
    m3 = make_test_method(mocker=mocker, method_name="method_sweep_val_3")
    mocker.patch.object(target=m3, attribute="execute", side_effect=method_sweep_val_3)
    m4 = make_test_method(mocker=mocker, method_name="method_sweep_val_4")
    mocker.patch.object(target=m4, attribute="execute", side_effect=method_sweep_val_4)

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    #
    # NOTE: the pipeline object is only needed for providing the loader, and the methods needed
    # for the purpose of this test are only the ones in the sweep stage, which is why the
    # pipeline and stages object both have only partial pipeline information
    pipeline = Pipeline(loader=loader, methods=[])
    stages = Stages(
        before_sweep=[],
        sweep=[m1, m2, m3, m4],
        after_sweep=[],
    )

    runner = ParamSweepRunner(pipeline=pipeline, stages=stages)
    runner.prepare()
    runner.execute_sweep()

    NO_OF_SWEEPS = 4
    EXPECTED_BLOCK_SHAPE = (
        PREVIEWED_SLICES_SHAPE[0],
        NO_OF_SWEEPS,
        PREVIEWED_SLICES_SHAPE[2],
    )
    assert runner.block.data.shape == EXPECTED_BLOCK_SHAPE
    expected_data = np.ones(EXPECTED_BLOCK_SHAPE, dtype=np.float32)
    expected_data[:, 0, :] *= 2
    expected_data[:, 1, :] *= 3
    expected_data[:, 2, :] *= 5
    expected_data[:, 3, :] *= 7
    np.testing.assert_array_equal(runner.block.data, expected_data)


def test_execute_modifies_block(mocker: MockerFixture):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.ones(PREVIEWED_SLICES_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)

    # Define two dummy methods to be in the before and sweep stages in the runner
    def dummy_method_1(block):
        block.data = block.data * 2
        return block

    def dummy_method_2(block):
        block.data = block.data * 3
        return block

    # Define four dummy functions that are representing a single method configured with
    # different parameter values
    def method_sweep_val_1(block):
        block.data = block.data * 2
        return block

    def method_sweep_val_2(block):
        block.data = block.data * 3
        return block

    def method_sweep_val_3(block):
        block.data = block.data * 5
        return block

    def method_sweep_val_4(block):
        block.data = block.data * 7
        return block

    # Create mock method wrapper objects and patch the `execute()` method on all of them to use
    # the dummy method functions defined above
    #
    # "Before sweep" stage method wrappers (2)
    m1 = make_test_method(mocker=mocker, method_name="method_1")
    mocker.patch.object(target=m1, attribute="execute", side_effect=dummy_method_1)
    m2 = make_test_method(mocker=mocker, method_name="method_2")
    mocker.patch.object(target=m2, attribute="execute", side_effect=dummy_method_2)
    # Sweep stage method wrappers (4)
    m3 = make_test_method(mocker=mocker, method_name="method_sweep_val_1")
    mocker.patch.object(target=m3, attribute="execute", side_effect=method_sweep_val_1)
    m4 = make_test_method(mocker=mocker, method_name="method_sweep_val_2")
    mocker.patch.object(target=m4, attribute="execute", side_effect=method_sweep_val_2)
    m5 = make_test_method(mocker=mocker, method_name="method_sweep_val_3")
    mocker.patch.object(target=m5, attribute="execute", side_effect=method_sweep_val_3)
    m6 = make_test_method(mocker=mocker, method_name="method_sweep_val_4")
    mocker.patch.object(target=m6, attribute="execute", side_effect=method_sweep_val_4)
    # "After sweep" stage method wrappers (2)
    m7 = make_test_method(mocker=mocker, method_name="method_7")
    mocker.patch.object(target=m7, attribute="execute", side_effect=dummy_method_1)
    m8 = make_test_method(mocker=mocker, method_name="method_8")
    mocker.patch.object(target=m8, attribute="execute", side_effect=dummy_method_2)

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    pipeline = Pipeline(loader=loader, methods=[m1, m2, m3, m4, m5, m6, m6, m7, m8])
    stages = Stages(
        before_sweep=[m1, m2],
        sweep=[m3, m4, m5, m6],
        after_sweep=[m7, m8],
    )

    runner = ParamSweepRunner(pipeline=pipeline, stages=stages)
    runner.execute()

    NO_OF_SWEEPS = 4
    EXPECTED_BLOCK_SHAPE = (
        PREVIEWED_SLICES_SHAPE[0],
        NO_OF_SWEEPS,
        PREVIEWED_SLICES_SHAPE[2],
    )
    assert runner.block.data.shape == EXPECTED_BLOCK_SHAPE
    BEFORE_SWEEP_MULT_FACTOR = AFTER_SWEEP_MULT_FACTOR = 2 * 3
    NON_SWEEP_STAGES_MULT_FACTOR = BEFORE_SWEEP_MULT_FACTOR * AFTER_SWEEP_MULT_FACTOR
    expected_data = np.ones(EXPECTED_BLOCK_SHAPE, dtype=np.float32)
    expected_data[:, 0, :] *= NON_SWEEP_STAGES_MULT_FACTOR * 2
    expected_data[:, 1, :] *= NON_SWEEP_STAGES_MULT_FACTOR * 3
    expected_data[:, 2, :] *= NON_SWEEP_STAGES_MULT_FACTOR * 5
    expected_data[:, 3, :] *= NON_SWEEP_STAGES_MULT_FACTOR * 7
    np.testing.assert_array_equal(runner.block.data, expected_data)


def test_execute_sweep_stage_method_params_updated_from_side_outputs(
    mocker: MockerFixture,
):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.ones(PREVIEWED_SLICES_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)

    # Define side output that dummy method in stage will require
    SIDE_OUTPUT_LABEL = "some_side_output"
    side_outputs = {SIDE_OUTPUT_LABEL: 5}
    side_output_manager = SideOutputManager()
    side_output_manager.append(side_outputs)

    # Define mock module + mock method function that will be imported by the method wrappers
    # that represent the parameter sweep.
    #
    # The mock method function asserts that the param value it's given is the value of the side
    # output that it requires.
    class FakeModule:
        def sweep_method(data, some_side_output) -> np.ndarray:  # type: ignore
            assert some_side_output == side_outputs[SIDE_OUTPUT_LABEL]
            return np.empty(PREVIEWED_SLICES_SHAPE, dtype=np.float32)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    sweep_wrappers = []
    NO_OF_SWEEP_VALS = 5
    for _ in range(NO_OF_SWEEP_VALS):
        method_wrapper = make_method_wrapper(
            method_repository=make_mock_repo(mocker),
            module_path="",
            method_name="sweep_method",
            comm=MPI.COMM_WORLD,
        )
        sweep_wrappers.append(method_wrapper)

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    #
    # NOTE: the pipeline object is only needed for providing the loader, and the methods needed
    # for the purpose of this test are only the ones in the sweep stage, which is why the
    # pipeline and stages object both have only partial pipeline information
    pipeline = Pipeline(loader=loader, methods=[])
    stages = Stages(
        before_sweep=[],
        sweep=sweep_wrappers,
        after_sweep=[],
    )
    runner = ParamSweepRunner(
        pipeline=pipeline, stages=stages, side_output_manager=side_output_manager
    )
    runner.prepare()
    runner.execute_sweep()


def test_execute_sweep_stage_method_produces_side_output_raises_error(
    mocker: MockerFixture,
):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 3, 160)
    data = np.ones(PREVIEWED_SLICES_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(PREVIEWED_SLICES_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )
    loader = make_test_loader(mocker, block=block)

    # Define side output that dummy method in sweep stage will produce
    SIDE_OUTPUT_LABEL = "some_side_output"
    side_outputs = {SIDE_OUTPUT_LABEL: 5}

    # Define mock method wrapper to produce the above side output in the sweep stage
    #
    # NOTE: Even though a sweep stage should generally have more than one method wrapper in it,
    # this test is for checking that a wrapper in the sweep stage producing a side output
    # causes an error to be raised. If even one method wrapper in the sweep stage produces a
    # side output, an error shoudl be raised, so only one method wrapper in the sweep stage is
    # needed to fulfill the purpose of the test.
    mock_wrapper = make_test_method(mocker)
    mocker.patch.object(
        target=mock_wrapper, attribute="get_side_output", return_value=side_outputs
    )

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    #
    # NOTE: the pipeline object is only needed for providing the loader, and the methods needed
    # for the purpose of this test are only the ones in the sweep stage, which is why the
    # pipeline and stages object both have only partial pipeline information
    pipeline = Pipeline(loader=loader, methods=[])
    stages = Stages(
        before_sweep=[],
        sweep=[mock_wrapper],
        after_sweep=[],
    )
    runner = ParamSweepRunner(pipeline=pipeline, stages=stages)
    runner.prepare()

    with pytest.raises(ValueError) as e:
        runner.execute_sweep()
    assert (
        "Producing a side output is not supported in parameter sweep methods"
        in str(e)
    )
