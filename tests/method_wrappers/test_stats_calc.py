import pytest
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.stats_calc import StatsCalcWrapper
from httomo.runner.dataset import DataSetBlock
from ..testing_utils import make_mock_preview_config, make_mock_repo

from mpi4py import MPI
from pytest_mock import MockerFixture
from httomo.utils import gpu_enabled, xp


def test_calculate_stats(mocker: MockerFixture, dummy_block: DataSetBlock):
    class FakeModule:
        def calculate_stats(data, comm):
            # outputs min/max/sum/total_elements
            return (1.2, 3.1, 42.0, 10)

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_SELF,
        make_mock_preview_config(mocker),
        output_mapping={"glob_stats": "glob_stats"},
    )
    assert isinstance(wrp, StatsCalcWrapper)
    wrp.execute(dummy_block)

    assert wrp.get_side_output() == {
        "glob_stats": (1.2, 3.1, 4.2, 10),  # computes mean (sum/total)
    }


def test_calculate_stats_supports_blockwise(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
    class FakeModule:
        def calculate_stats(data, comm):
            # outputs min/max/sum/total_elements
            return (1.2, 3.1, 42.0, 10)

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_SELF,
        make_mock_preview_config(mocker),
        output_mapping={"glob_stats": "glob_stats"},
    )
    assert isinstance(wrp, StatsCalcWrapper)
    # execute with 2 blocks
    b1 = DataSetBlock(
        data=dummy_block.data[:2, :, :],
        aux_data=dummy_block.aux_data,
        slicing_dim=0,
        chunk_shape=dummy_block.chunk_shape,
        global_shape=dummy_block.global_shape,
        block_start=0,
        chunk_start=0,
    )
    b2 = DataSetBlock(
        data=dummy_block.data[2:, :, :],
        aux_data=dummy_block.aux_data,
        slicing_dim=0,
        chunk_shape=dummy_block.chunk_shape,
        global_shape=dummy_block.global_shape,
        block_start=2,
        chunk_start=0,
    )
    wrp.execute(b1)
    wrp.execute(b2)

    assert wrp.get_side_output() == {
        "glob_stats": (
            1.2,
            3.1,
            4.2,
            20,
        ),  # computes mean (sum/total), accumulates elements
    }


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_calculate_stats_2_processes(mocker: MockerFixture, dummy_block: DataSetBlock):
    class FakeModule:
        def calculate_stats(data, comm):
            # outputs min/max/sum/total_elements
            if comm.rank == 0:
                return (1.2, 3.1, 42.0, 10)
            else:
                return (1.1, 3.5, 40.0, 9)

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        output_mapping={"glob_stats": "glob_stats"},
    )
    wrp.execute(dummy_block)

    assert wrp.get_side_output() == {
        "glob_stats": (1.1, 3.5, (42.0 + 40.0) / 19, 19),  # computes mean (sum/total)
    }


@pytest.mark.parametrize("gpu", [False, True], ids=["CPU-input", "GPU-input"])
def test_calculate_stats_uses_gpu_if_available(
    mocker: MockerFixture, dummy_block: DataSetBlock, gpu: bool
):
    if gpu and not gpu_enabled:
        pytest.skip("No GPU available")

    class FakeModule:
        def calculate_stats(data, comm):
            # regardless of dataset input, we want device data if gpu enabled
            if gpu_enabled:
                assert data.device != "cpu"
            else:
                assert data.device == "cpu"
            return (1.2, 3.1, 42.0, 10)

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_SELF,
        make_mock_preview_config(mocker),
        output_mapping={"glob_stats": "glob_stats"},
    )

    if gpu is True:
        dummy_block.to_gpu()

    res = wrp.execute(dummy_block)

    assert res.is_gpu == gpu
