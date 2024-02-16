import pytest
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.stats_calc import StatsCalcWrapper
from httomo.runner.dataset import DataSet
from ..testing_utils import make_mock_repo

from mpi4py import MPI
from pytest_mock import MockerFixture
from httomo.utils import gpu_enabled, xp


def test_calculate_stats(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def calculate_stats(data, comm):
            # outputs min/max/sum/total_elements
            return (1.2, 3.1, 42.0, 10)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_SELF,
        output_mapping={"glob_stats": "glob_stats"},
    )
    assert isinstance(wrp, StatsCalcWrapper)
    wrp.execute(dummy_dataset.make_block(0))

    assert wrp.get_side_output() == {
        "glob_stats": (1.2, 3.1, 4.2, 10),  # computes mean (sum/total)
    }


def test_calculate_stats_supports_blockwise(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def calculate_stats(data, comm):
            # outputs min/max/sum/total_elements
            return (1.2, 3.1, 42.0, 10)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_SELF,
        output_mapping={"glob_stats": "glob_stats"},
    )
    assert isinstance(wrp, StatsCalcWrapper)
    # execute with 2 blocks
    wrp.execute(dummy_dataset.make_block(0, 0, dummy_dataset.shape[0] // 2))
    wrp.execute(dummy_dataset.make_block(0, dummy_dataset.shape[0] // 2))

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
def test_calculate_stats_2_processes(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def calculate_stats(data, comm):
            # outputs min/max/sum/total_elements
            if comm.rank == 0:
                return (1.2, 3.1, 42.0, 10)
            else:
                return (1.1, 3.5, 40.0, 9)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_WORLD,
        output_mapping={"glob_stats": "glob_stats"},
    )
    wrp.execute(dummy_dataset.make_block(0))

    assert wrp.get_side_output() == {
        "glob_stats": (1.1, 3.5, (42.0 + 40.0) / 19, 19),  # computes mean (sum/total)
    }


@pytest.mark.parametrize("gpu", [False, True], ids=["CPU-input", "GPU-input"])
def test_calculate_stats_uses_gpu_if_available(
    mocker: MockerFixture, dummy_dataset: DataSet, gpu: bool
):

    if gpu and not gpu_enabled:
        pytest.skip("No GPU available")

    class FakeModule:
        def calculate_stats(data, comm):
            # regardless of dataset input, we want device data if gpu enabled
            if gpu_enabled:
                assert getattr(data, "device", None) is not None
            else:
                assert getattr(data, "device", None) is None
            return (1.2, 3.1, 42.0, 10)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_SELF,
        output_mapping={"glob_stats": "glob_stats"},
    )

    if gpu is True:
        dummy_dataset.to_gpu()

    res = wrp.execute(dummy_dataset.make_block(0))

    assert res.is_gpu == gpu
