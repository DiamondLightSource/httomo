from unittest import mock

import pytest
from pytest_mock import MockerFixture
from typing import Literal

import numpy as np
from mpi4py import MPI

from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.method_wrappers.images import ImagesWrapper
from httomo.method_wrappers.save_intermediate import SaveIntermediateFilesWrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.pipeline import Pipeline
from httomo.preview import PreviewConfig, PreviewDimConfig
from httomo.sweep_runner.param_sweep_block import ParamSweepBlock
from httomo.sweep_runner.param_sweep_runner import ParamSweepRunner
from httomo.sweep_runner.side_output_manager import SideOutputManager
from httomo.sweep_runner.stages import NonSweepStage, Stages, SweepStage
from tests.testing_utils import (
    make_mock_preview_config,
    make_mock_repo,
    make_test_loader,
    make_test_method,
)
from httomo.runner.loader import LoaderInterface
from httomo.runner.dataset_store_interfaces import DataSetSource

from httomo_backends.methods_database.query import Pattern


def test_raises_error_if_no_sweep_detected(mocker: MockerFixture):
    loader = make_test_loader(mocker)
    method = make_test_method(mocker)
    pipeline = Pipeline(loader=loader, methods=[method])

    with pytest.raises(ValueError) as e:
        _ = ParamSweepRunner(pipeline, MPI.COMM_WORLD)

    assert "No parameter sweep detected in pipeline" in str(e)


def test_raises_error_if_multiple_sweeps_detected(mocker: MockerFixture):
    loader = make_test_loader(mocker)
    m1 = make_test_method(mocker, param_1=(0, 1, 2, 3))
    m2 = make_test_method(mocker, param_2=(4, 5, 6, 7))
    pipeline = Pipeline(loader=loader, methods=[m1, m2])

    with pytest.raises(ValueError) as e:
        _ = ParamSweepRunner(pipeline, MPI.COMM_WORLD)

    assert "Parameter sweep over more than one parameter detected in pipeline" in str(e)


def test_determine_stages_produces_correct_stages(mocker: MockerFixture):
    SWEEP_VALS = (0, 1, 2, 3)
    loader = make_test_loader(mocker)
    m1 = make_test_method(mocker)
    sweep_wrapper = make_test_method(mocker, param_1=SWEEP_VALS)
    m3 = make_test_method(mocker)
    pipeline = Pipeline(loader=loader, methods=[m1, sweep_wrapper, m3])
    expected_stages = Stages(
        before_sweep=NonSweepStage([m1]),
        sweep=SweepStage(method=sweep_wrapper, param_name="param_1", values=SWEEP_VALS),
        after_sweep=NonSweepStage([m3]),
    )
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    stages = runner.determine_stages()
    assert stages == expected_stages


def test_without_prepare_block_property_raises_error(mocker: MockerFixture):
    loader = make_test_loader(mocker)
    m1 = make_test_method(mocker)
    sweep_wrapper = make_test_method(mocker, param_1=(0,))
    pipeline = Pipeline(loader=loader, methods=[m1, sweep_wrapper])
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    with pytest.raises(ValueError) as e:
        runner.block
    assert "Block from input data has not yet been loaded" in str(e)


def test_after_prepare_block_attr_contains_data(mocker: MockerFixture):
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, 10, 160)
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
    m1 = make_test_method(mocker)
    sweep_wrapper = make_test_method(mocker, param_1=(0,))
    pipeline = Pipeline(loader=loader, methods=[m1, sweep_wrapper])
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    runner.prepare()
    assert runner.block is not None
    np.testing.assert_array_equal(runner.block.data, data)


def tests_preview_modifier_paganin_tomopy(mocker: MockerFixture):
    SINO_SLICES = 100
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, SINO_SLICES, 160)
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
    preview = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[0]),
        detector_y=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[1]),
        detector_x=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[2]),
    )

    class FakeModule:
        def paganin_filter_tomopy(data: np.ndarray, alpha: float, pixel_size: float, energy: float, dist: float):  # type: ignore
            return data * alpha

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface,
        instance=True,
        pattern=Pattern.all,
        method_name="testloader",
        reslice=False,
        preview=preview,
    )

    def mock_make_data_source(padding) -> DataSetSource:
        ret = mocker.create_autospec(
            DataSetSource,
            global_shape=block.global_shape,
            dtype=block.data.dtype,
            chunk_shape=block.chunk_shape,
            chunk_index=block.chunk_index,
            slicing_dim=1 if loader.pattern == Pattern.sinogram else 0,
            aux_data=block.aux_data,
            preview=preview,
        )
        type(ret).raw_shape = mock.PropertyMock(return_value=GLOBAL_SHAPE)
        slicing_dim: Literal[0, 1, 2] = 0
        mocker.patch.object(
            ret,
            "read_block",
            side_effect=lambda start, length: DataSetBlock(
                data=block.data[start : start + length, :, :],
                aux_data=block.aux_data,
                global_shape=block.global_shape,
                chunk_shape=block.chunk_shape,
                slicing_dim=slicing_dim,
                block_start=start,
                chunk_start=block.chunk_index[slicing_dim],
            ),
        )
        return ret

    mocker.patch.object(
        loader,
        "make_data_source",
        side_effect=mock_make_data_source,
    )

    # Create sweep method wrapper, passing the sweep values for the param to sweep over
    SWEEP_VALUES = (10, 20)
    paganin_params = {
        "alpha": SWEEP_VALUES,
        "pixel_size": 0.0001,
        "energy": 53,
        "dist": 50,
    }
    sweep_method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="mocked_module_path.corr",
        method_name="paganin_filter_tomopy",
        comm=MPI.COMM_WORLD,
        preview_config=make_mock_preview_config(mocker),
        save_result=None,
        output_mapping={},
        **paganin_params,
    )

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    pipeline = Pipeline(loader=loader, methods=[sweep_method_wrapper])
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    runner.modify_loader_preview()

    expected_preview_config = PreviewDimConfig(start=40, stop=60)
    assert loader.preview.detector_y == expected_preview_config


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

    # Define sweep wrapper to to provide the runner the ability to distinguish between "before
    # sweep" and "after sweep" stages
    sweep_wrapper = make_test_method(mocker, param_1=(0,))

    # Define pipeline object that produces the "before sweep" and "after sweep" stages desired
    # for this parametrised test
    if non_sweep_stage == "before":
        pipeline = Pipeline(loader=loader, methods=[m1, m2, sweep_wrapper])
    else:
        pipeline = Pipeline(loader=loader, methods=[sweep_wrapper, m1, m2])

    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    runner.prepare()
    if non_sweep_stage == "before":
        runner.execute_before_sweep()
    else:
        runner.execute_after_sweep()

    # Inspect the block data after the non-sweep stage has completed, asserting that the data
    # reflects what the combination of two dummy methods in that stage should produce
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

    # Define sweep wrapper to to provide the runner the ability to distinguish between "before
    # sweep" and "after sweep" stages
    sweep_wrapper = make_test_method(mocker, param_1=(0,))

    # Define pipeline object that produces the "before sweep" and "after sweep" stages desired
    # for this parametrised test
    if non_sweep_stage == "before":
        pipeline = Pipeline(loader=loader, methods=[mock_wrapper, sweep_wrapper])
    else:
        pipeline = Pipeline(loader=loader, methods=[sweep_wrapper, mock_wrapper])

    # Pass in an empty side output manager object into the runner. The execution of the method
    # should add the side output produced by it to the side output manager.
    side_output_manager = SideOutputManager()
    runner = ParamSweepRunner(
        pipeline=pipeline, comm=MPI.COMM_WORLD, side_output_manager=side_output_manager
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

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="",
        method_name="method_1",
        comm=MPI.COMM_WORLD,
        preview_config=make_mock_preview_config(mocker),
    )

    # Define sweep wrapper to to provide the runner the ability to distinguish between "before
    # sweep" and "after sweep" stages
    sweep_wrapper = make_test_method(mocker, param_1=(0,))

    # Define pipeline object that produces the "before sweep" and "after sweep" stages desired
    # for this parametrised test
    if non_sweep_stage == "before":
        pipeline = Pipeline(loader=loader, methods=[method_wrapper, sweep_wrapper])
    else:
        pipeline = Pipeline(loader=loader, methods=[sweep_wrapper, method_wrapper])

    runner = ParamSweepRunner(
        pipeline=pipeline, comm=MPI.COMM_WORLD, side_output_manager=side_output_manager
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

    # Define a dummy method function that the method wrapper will be patched to import. When
    # executing the sweep stage, different values will be passed to it, which should influence
    # the data in the block produced each time the method wrapper is executed in the
    # sweep.
    class FakeModule:
        def sweep_method(data: np.ndarray, param_1: int):  # type: ignore
            return data * param_1

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    # Create sweep method wrapper, passing the sweep values for the param to sweep over
    NO_OF_SWEEPS = 4
    SWEEP_VALUES = (2, 3, 5, 7)
    sweep_method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="mocked_module_path.corr",
        method_name="sweep_method",
        comm=MPI.COMM_WORLD,
        preview_config=make_mock_preview_config(mocker),
        param_1=SWEEP_VALUES,
    )

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    pipeline = Pipeline(loader=loader, methods=[sweep_method_wrapper])
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    runner.prepare()
    runner.execute_sweep()

    EXPECTED_BLOCK_SHAPE = (
        PREVIEWED_SLICES_SHAPE[0],
        NO_OF_SWEEPS,
        PREVIEWED_SLICES_SHAPE[2],
    )
    assert runner.block.data.shape == EXPECTED_BLOCK_SHAPE
    expected_data = np.ones(EXPECTED_BLOCK_SHAPE, dtype=np.float32)
    for sino_slice_idx, multiplier in enumerate(SWEEP_VALUES):
        expected_data[:, sino_slice_idx, :] *= multiplier
    np.testing.assert_array_equal(runner.block.data, expected_data)


def tests_modify_loader_preview_raises_error_if_too_few_sino_slices(
    mocker: MockerFixture,
):
    TOO_FEW_SINO_SLICES = 3
    GLOBAL_SHAPE = PREVIEWED_SLICES_SHAPE = (180, TOO_FEW_SINO_SLICES, 160)
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
    preview = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[0]),
        detector_y=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[1]),
        detector_x=PreviewDimConfig(start=0, stop=PREVIEWED_SLICES_SHAPE[2]),
    )

    # loader = make_test_loader(mocker, preview=preview, block=block)
    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface,
        instance=True,
        pattern=Pattern.all,
        method_name="testloader",
        reslice=False,
        preview=preview,
    )

    def mock_make_data_source(padding) -> DataSetSource:
        ret = mocker.create_autospec(
            DataSetSource,
            global_shape=block.global_shape,
            dtype=block.data.dtype,
            chunk_shape=block.chunk_shape,
            chunk_index=block.chunk_index,
            slicing_dim=1 if loader.pattern == Pattern.sinogram else 0,
            aux_data=block.aux_data,
            preview=preview,
        )
        type(ret).raw_shape = mock.PropertyMock(return_value=GLOBAL_SHAPE)
        slicing_dim: Literal[0, 1, 2] = 0
        mocker.patch.object(
            ret,
            "read_block",
            side_effect=lambda start, length: DataSetBlock(
                data=block.data[start : start + length, :, :],
                aux_data=block.aux_data,
                global_shape=block.global_shape,
                chunk_shape=block.chunk_shape,
                slicing_dim=slicing_dim,
                block_start=start,
                chunk_start=block.chunk_index[slicing_dim],
            ),
        )
        return ret

    mocker.patch.object(
        loader,
        "make_data_source",
        side_effect=mock_make_data_source,
    )

    m1 = make_test_method(mocker)
    sweep_wrapper = make_test_method(mocker, param_1=(0,))
    pipeline = Pipeline(loader=loader, methods=[m1, sweep_wrapper])
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)

    with pytest.raises(ValueError) as e:
        runner.modify_loader_preview()

    err_str = "The size of the raw data  3 is smaller than it is required by the sweep run - 5. Please consider using a larger input dataset with the sweep run."
    assert err_str in str(e)


def test_execute_modifies_block_extend_preview(mocker: MockerFixture):
    # Define dummy block for loader to provide
    GLOBAL_SHAPE = (180, 15, 160)
    data = np.ones(GLOBAL_SHAPE, dtype=np.float32)
    aux_data = AuxiliaryData(np.ones(GLOBAL_SHAPE[0], dtype=np.float32))
    block = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        global_shape=GLOBAL_SHAPE,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        block_start=0,
    )

    # For non-sweep stages, the `execute()` method on the mock wrappers will be patched. For
    # the sweep stage, the method function that the wrapper imports will be patched.
    #
    # This is because for non-sweep stages, all that is needed is to see the block data change.
    # For the sweep stage, it needs to be confirmed that the sweep values are actually being
    # used in the execution of the sweep. The easiest way to see this is for the parameter
    # value to be used as a simple multiplier in the method function that the wrapper executes,
    # which would cause the block data to change according to the given sweep values.

    # Define two dummy methods to be in the before and sweep stages in the runner
    def dummy_method_1(block):
        block.data = block.data * 2
        return block

    def dummy_method_2(block):
        block.data = block.data * 3
        return block

    # Define a dummy method function that the method wrapper will be patched to import. When
    # executing the sweep stage, different values will be passed to it, which should influence
    # the data in the block produced each time the method wrapper is executed in the
    # sweep.
    class FakeModule:
        def sweep_method(data: np.ndarray, param_1: int):  # type: ignore
            return data * param_1

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    preview = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=GLOBAL_SHAPE[0]),
        detector_y=PreviewDimConfig(start=7, stop=10),
        detector_x=PreviewDimConfig(start=0, stop=GLOBAL_SHAPE[2]),
    )

    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface,
        instance=True,
        pattern=Pattern.all,
        method_name="testloader",
        reslice=False,
        preview=preview,
    )

    def mock_make_data_source(padding) -> DataSetSource:
        ret = mocker.create_autospec(
            DataSetSource,
            global_shape=block.global_shape,
            dtype=block.data.dtype,
            chunk_shape=block.chunk_shape,
            chunk_index=block.chunk_index,
            slicing_dim=1 if loader.pattern == Pattern.sinogram else 0,
            aux_data=block.aux_data,
            preview=preview,
        )
        type(ret).raw_shape = mock.PropertyMock(return_value=GLOBAL_SHAPE)
        slicing_dim: Literal[0, 1, 2] = 0
        mocker.patch.object(
            ret,
            "read_block",
            side_effect=lambda start, length: DataSetBlock(
                data=block.data[start : start + length, :, :],
                aux_data=block.aux_data,
                global_shape=block.global_shape,
                chunk_shape=block.chunk_shape,
                slicing_dim=slicing_dim,
                block_start=start,
                chunk_start=block.chunk_index[slicing_dim],
            ),
        )
        return ret

    mocker.patch.object(
        loader,
        "make_data_source",
        side_effect=mock_make_data_source,
    )
    # Create sweep method wrapper, passing the sweep values for the param to sweep over
    NO_OF_SWEEPS = 4
    SWEEP_VALUES = (2, 3, 5, 7)
    sweep_method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="mocked_module_path.corr",
        method_name="sweep_method",
        comm=MPI.COMM_WORLD,
        preview_config=make_mock_preview_config(mocker),
        param_1=SWEEP_VALUES,
    )

    # Create mock method wrapper objects to be used in the non-sweep stages and patch the
    # `execute()` method on all of them to use the dummy method functions defined above
    #
    # "Before sweep" stage method wrappers (2)
    m1 = make_test_method(mocker=mocker, method_name="method_1")
    mocker.patch.object(target=m1, attribute="execute", side_effect=dummy_method_1)
    m2 = make_test_method(mocker=mocker, method_name="method_2")
    mocker.patch.object(target=m2, attribute="execute", side_effect=dummy_method_2)
    # "After sweep" stage method wrappers (2)
    m4 = make_test_method(mocker=mocker, method_name="method_4")
    mocker.patch.object(target=m4, attribute="execute", side_effect=dummy_method_1)
    m5 = make_test_method(mocker=mocker, method_name="method_5")
    mocker.patch.object(target=m5, attribute="execute", side_effect=dummy_method_2)

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    pipeline = Pipeline(loader=loader, methods=[m1, m2, sweep_method_wrapper, m4, m5])
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    runner.execute()

    NO_OF_SWEEPS = 4
    EXPECTED_BLOCK_SHAPE = (
        GLOBAL_SHAPE[0],
        NO_OF_SWEEPS,
        GLOBAL_SHAPE[2],
    )
    assert runner.block.data.shape == EXPECTED_BLOCK_SHAPE

    expected_preview_config = PreviewDimConfig(start=6, stop=11)
    assert loader.preview.detector_y == expected_preview_config
    BEFORE_SWEEP_MULT_FACTOR = AFTER_SWEEP_MULT_FACTOR = 2 * 3
    NON_SWEEP_STAGES_MULT_FACTOR = BEFORE_SWEEP_MULT_FACTOR * AFTER_SWEEP_MULT_FACTOR
    expected_data = np.ones(EXPECTED_BLOCK_SHAPE, dtype=np.float32)
    for sino_slice_idx, multiplier in enumerate(SWEEP_VALUES):
        expected_data[:, sino_slice_idx, :] *= NON_SWEEP_STAGES_MULT_FACTOR * multiplier
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

    # Define mock module + mock method function that will be imported by the method wrapper
    # that represents the parameter sweep.
    #
    # The mock method function asserts that the param value it's given is the value of the side
    # output that it requires.
    class FakeModule:
        def sweep_method(data, param_1, some_side_output) -> np.ndarray:  # type: ignore
            assert some_side_output == side_outputs[SIDE_OUTPUT_LABEL]
            return np.empty(PREVIEWED_SLICES_SHAPE, dtype=np.float32)

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    # Create sweep method wrapper, passing the sweep values for the param to sweep over
    SWEEP_VALUES = (2, 3, 5, 7)
    sweep_method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="mocked_module_path.corr",
        method_name="sweep_method",
        comm=MPI.COMM_WORLD,
        preview_config=make_mock_preview_config(mocker),
        param_1=SWEEP_VALUES,
    )

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    pipeline = Pipeline(loader=loader, methods=[sweep_method_wrapper])
    runner = ParamSweepRunner(
        pipeline=pipeline, comm=MPI.COMM_WORLD, side_output_manager=side_output_manager
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
    # NOTE: Even though a sweep stage should generally have more than one sweep value in it,
    # this test is for checking that a wrapper execution in the sweep stage producing a side
    # output causes an error to be raised. If even one method wrapper execution in the sweep
    # stage produces a side output, an error should be raised, so only one sweep value in the
    # sweep stage is needed to fulfill the purpose of the test.
    mock_wrapper = make_test_method(mocker, param_1=(1,))
    mocker.patch.object(
        target=mock_wrapper, attribute="get_side_output", return_value=side_outputs
    )

    # Define pipeline, stages, runner objects, and execute the methods in the sweep stage
    pipeline = Pipeline(loader=loader, methods=[mock_wrapper])
    runner = ParamSweepRunner(pipeline, MPI.COMM_WORLD)
    runner.prepare()

    with pytest.raises(ValueError) as e:
        runner.execute_sweep()
    assert "Producing a side output is not supported in parameter sweep methods" in str(
        e
    )


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_execute_sweep_stage_two_procs_correct_sweep_val_distribution(
    mocker: MockerFixture,
):
    # Define communicator objects
    global_comm = MPI.COMM_WORLD
    method_wrapper_comm = MPI.COMM_SELF

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

    # Define a dummy method function that the method wrapper will be patched to import. When
    # executing the sweep stage, each rank should use a different subset of the given sweep
    # values.
    class FakeModule:
        def sweep_method(data: np.ndarray, param_1: int):  # type: ignore
            pass

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    # Cretae spy on dummy method function to later assert that, for each rank, it has been
    # called with the expected subset of sweep values
    method_function_spy = mocker.patch.object(
        target=FakeModule,
        attribute="sweep_method",
        return_value=data,
        autospec=FakeModule.sweep_method,  # type: ignore
    )

    # Create sweep method wrapper, passing the sweep values for the param to sweep over
    SWEEP_VALUES = tuple(list(range(10)))
    sweep_method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="mocked_module_path.corr",
        method_name="sweep_method",
        comm=method_wrapper_comm,
        preview_config=make_mock_preview_config(mocker),
        param_1=SWEEP_VALUES,
    )

    # Define pipeline + runner objects, and execute the sweep stage
    pipeline = Pipeline(loader=loader, methods=[sweep_method_wrapper])
    runner = ParamSweepRunner(pipeline, global_comm)
    runner.prepare()
    runner.execute_sweep()

    # Rank 0 should execute the first five sweep values, and rank 1 should execute the last
    # five sweep values
    if global_comm.rank == 0:
        expected_sweep_vals = list(SWEEP_VALUES)[:5]
    else:
        expected_sweep_vals = list(SWEEP_VALUES)[5:]

    # Define a list of mock call objects that contain the expected sweep values the dummy
    # method function should be called with
    expected_mock_calls = [mock.call(args=data, param_1=i) for i in expected_sweep_vals]

    assert len(method_function_spy.mock_calls) == len(expected_mock_calls)

    # For each mock call object from the dummy method function executed in the sweep, assert
    # that:
    # - the args are the same (should be empty, no positional args are given, even the numpy
    # array for the method function is passed as a kwarg)
    # - the kwargs dict contains a `data` key and a `param_1` key (representing the two kwargs
    # passed to the dummy method function)
    for mock_call, expected_mock_call in zip(
        method_function_spy.mock_calls, expected_mock_calls
    ):
        assert mock_call.args == expected_mock_call.args
        assert len(mock_call.kwargs) == 2
        assert "data" in mock_call.kwargs.keys()
        assert "param_1" in mock_call.kwargs.keys()
        assert mock_call.kwargs["param_1"] == expected_mock_call.kwargs["param_1"]


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_execute_sweep_stage_two_procs_correct_middle_slices_in_block(
    mocker: MockerFixture,
):
    # Define communicator objects
    global_comm = MPI.COMM_WORLD
    method_wrapper_comm = MPI.COMM_SELF

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

    # Define a dummy method function that the method wrapper will be patched to import. When
    # executing the sweep stage, each rank should use a different subset of the given sweep
    # values.
    class FakeModule:
        def sweep_method(data: np.ndarray, param_1: int):  # type: ignore
            return data * param_1

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    # Create sweep method wrapper, passing the sweep values for the param to sweep over
    SWEEP_VALUES = tuple(list(range(10)))
    sweep_method_wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path="mocked_module_path.corr",
        method_name="sweep_method",
        comm=method_wrapper_comm,
        preview_config=make_mock_preview_config(mocker),
        param_1=SWEEP_VALUES,
    )

    # Define pipeline + runner objects, and execute the sweep stage
    pipeline = Pipeline(loader=loader, methods=[sweep_method_wrapper])
    runner = ParamSweepRunner(pipeline, global_comm)
    runner.prepare()
    runner.execute_sweep()

    # Verify that the middle slices contained in the block data for both processes are the
    # expected middle slices (based on the sweep values that each process was responsible for
    # executing)
    NO_OF_MIDDLE_SLICES_PER_PROCESS = 5
    middle_slices_shape = (
        GLOBAL_SHAPE[0],
        NO_OF_MIDDLE_SLICES_PER_PROCESS,
        GLOBAL_SHAPE[2],
    )
    expected_data = np.ones(middle_slices_shape, dtype=np.float32)

    if global_comm.rank == 0:
        sweep_values = SWEEP_VALUES[:5]
    else:
        sweep_values = SWEEP_VALUES[5:]

    for middle_slice_idx, val in enumerate(sweep_values):
        expected_data[:, middle_slice_idx, :] *= val

    np.testing.assert_array_equal(runner.block.data, expected_data)


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_execute_sweep_stage_two_procs_image_wrapper_gets_correct_offset(
    mocker: MockerFixture,
):
    # Define communicator objects
    global_comm = MPI.COMM_WORLD
    method_wrapper_comm = MPI.COMM_SELF

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

    # Define mock method wrappers
    SWEEP_VALUES = tuple(list(range(10)))
    sweep_method_wrapper = mocker.create_autospec(
        GenericMethodWrapper, instance=True, config_params={"sweep_param": SWEEP_VALUES}
    )
    before_sweep_wrapper = mocker.create_autospec(
        GenericMethodWrapper, instance=True, config_params={"param_1": 5}
    )
    after_sweep_wrapper_images = mocker.create_autospec(
        ImagesWrapper, instance=True, config_params={"param_2": 10}
    )

    # Define spies on the mock method wrapper's `__setitem__()` method, to be used later to
    # verify if `__setitem__()` has been called or not (and if so, with what args)
    sweep_wrapper_setitem_spy = mocker.patch.object(
        target=sweep_method_wrapper, attribute="__setitem__"
    )
    before_sweep_wrapper_setitem_spy = mocker.patch.object(
        target=before_sweep_wrapper, attribute="__setitem__"
    )
    after_sweep_wrapper_setitem_spy = mocker.patch.object(
        target=after_sweep_wrapper_images, attribute="__setitem__"
    )

    # Define pipeline and runner objects
    pipeline = Pipeline(
        loader=loader,
        methods=[
            before_sweep_wrapper,
            sweep_method_wrapper,
            after_sweep_wrapper_images,
        ],
    )
    _ = ParamSweepRunner(pipeline, global_comm)

    # Verify that:
    # - the mock images wrapper has the correct offset parameter value set, based on the rank
    # of the process
    # - the before and after sweep method wrappers don't have any parameter values being set
    expected_offset = 0 if global_comm.rank == 0 else len(SWEEP_VALUES) // 2
    after_sweep_wrapper_setitem_spy.assert_has_calls(
        [mock.call("offset", expected_offset)]
    )
    before_sweep_wrapper_setitem_spy.assert_not_called()
    sweep_wrapper_setitem_spy.assert_not_called()


def test_passes_minimum_block_length_to_intermediate_wrapper(mocker: MockerFixture):
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

    # The output of the different method wrappers doesn't matter for this test, so mock method
    # wrappers which return some predetermined instance of `ParamSweepBlock` is sufficient
    sweep_block = ParamSweepBlock(data=data, aux_data=aux_data, slicing_dim=0)

    # Define mock method wrapper whose output is intended to be saved
    method_wrapper = make_test_method(mocker=mocker, save_result=True)
    method_wrapper_spy = mocker.patch.object(method_wrapper, "append_config_params")
    mocker.patch.object(
        target=method_wrapper, attribute="execute", side_effect=[sweep_block]
    )

    # Define intermediate saver wrapper which is placed after the method whose output should be
    # saved
    saver_wrapper = mocker.create_autospec(
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
    mocker.patch.object(
        target=saver_wrapper, attribute="execute", side_effect=[sweep_block]
    )
    saver_wrapper_spy = mocker.patch.object(saver_wrapper, "append_config_params")

    # Define method wrapper for method which has a parameter sweep defined
    SWEEP_VALUES = (0,)
    sweep_wrapper = make_test_method(mocker, param_1=SWEEP_VALUES)
    mocker.patch.object(
        target=sweep_wrapper, attribute="execute", side_effect=[sweep_block]
    )
    sweep_wrapper_spy = mocker.patch.object(sweep_wrapper, "append_config_params")

    pipeline = Pipeline(
        loader=loader, methods=[method_wrapper, saver_wrapper, sweep_wrapper]
    )
    runner = ParamSweepRunner(pipeline=pipeline, comm=MPI.COMM_WORLD)
    runner.prepare()

    # The method wrapper before the saver wrapper shouldn't have `append_config_params()`
    # called at all, whereas the saver wrapper should have `append_config_params()` called once
    # with the expected `minimum_block_length` value
    runner.execute_before_sweep()
    method_wrapper_spy.assert_not_called()
    saver_wrapper_spy.assert_called_once_with({"minimum_block_length": GLOBAL_SHAPE[1]})

    # The `append_config_params()` method should only be called on the sweep method wrapper for
    # the parameter on which the sweep is being done on, not for the `minimum_block_length`
    # parameter
    runner.execute_sweep()
    sweep_wrapper_spy.assert_has_calls([mock.call({"param_1": SWEEP_VALUES[0]})])
