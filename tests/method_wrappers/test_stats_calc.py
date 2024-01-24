from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.stats_calc import StatsCalcWrapper
from httomo.runner.dataset import DataSet
from ..testing_utils import make_mock_repo


from mpi4py import MPI
from pytest_mock import MockerFixture


def test_stats_calculate_stats(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def calculate_stats(data):
            return data


    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.calculate_stats",
        "calculate_stats",
        MPI.COMM_WORLD,
        output_mapping={"glob_stats": "glob_stats"},
    )
    assert isinstance(wrp, StatsCalcWrapper)
    wrp.execute(dummy_dataset.make_block(0))

    assert wrp.get_side_output() == {
        "glob_stats": (1.0, 1.0, 1.0, 1000),
    }