from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.dezinging import DezingingWrapper
from httomo.runner.dataset import DataSet
from ..testing_utils import make_mock_repo


import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_dezinging(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def remove_outlier3d(x):
            return 2 * x

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.prep",
        "remove_outlier3d",
        MPI.COMM_WORLD,
    )
    assert isinstance(wrp, DezingingWrapper)
    dummy_dataset.unlock()
    dummy_dataset.data[:] = 1
    dummy_dataset.flats[:] = 3
    dummy_dataset.darks[:] = 4
    dummy_dataset.lock()

    block1 = dummy_dataset.make_block(0, 0, 2)
    newblock1 = wrp.execute(block1)

    # we double them all, so we expect all 3 to be twice the input
    np.testing.assert_array_equal(newblock1.data, 2)
    np.testing.assert_array_equal(newblock1.flats, 6)
    np.testing.assert_array_equal(newblock1.darks, 8)

    # it should only update the darks/flats once, but still apply to the data
    block2 = dummy_dataset.make_block(0, 2, 2)
    newblock2 = wrp.execute(block2)
    np.testing.assert_array_equal(newblock2.data, 2)
    np.testing.assert_array_equal(newblock2.flats, 6)
    np.testing.assert_array_equal(newblock2.darks, 8)

    # the flats, darks in original dataset should have changed by reference
    np.testing.assert_array_equal(dummy_dataset.flats, 6)
    np.testing.assert_array_equal(dummy_dataset.darks, 8)