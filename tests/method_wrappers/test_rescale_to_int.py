import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.rescale_to_int import RescaleToIntWrapper
from httomo.runner.dataset import DataSetBlock
from ..testing_utils import make_mock_preview_config, make_mock_repo


def test_transform_glob_stats_list_to_tuple(
    mocker: MockerFixture, dummy_block: DataSetBlock
):
    class FakeModule:
        def rescale_to_int(data, perc_range_min, perc_range_max, bits, glob_stats):
            assert isinstance(glob_stats, tuple)
            assert len(glob_stats) == len(PARAMS["glob_stats"])
            for list_value, tuple_value in zip(PARAMS["glob_stats"], glob_stats):
                assert list_value == pytest.approx(tuple_value)
            return data

    PARAMS = {
        "perc_range_min": 1.0,
        "perc_range_max": 99.0,
        "bits": 8,
        "glob_stats": [1, 2, 3, 4],
    }
    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.misc",
        "rescale_to_int",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        **PARAMS,
    )
    assert isinstance(wrp, RescaleToIntWrapper)
    wrp.execute(dummy_block)
