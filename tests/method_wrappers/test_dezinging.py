import math
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.dezinging import DezingingWrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from ..testing_utils import make_mock_repo


import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_dezinging(mocker: MockerFixture):
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

    GLOBAL_SHAPE = (10, 20, 30)
    darks = 4 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), dtype=np.float32)
    flats = 3 * np.ones((3, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), dtype=np.float32)
    aux_data = AuxiliaryData(
        angles=np.linspace(0, math.pi, GLOBAL_SHAPE[0], dtype=np.float32),
        darks=darks,
        flats=flats,
    )

    darks_setter_spy = mocker.spy(aux_data, "set_darks")
    flats_setter_spy = mocker.spy(aux_data, "set_flats")
    data = np.ones(GLOBAL_SHAPE, dtype=np.float32)

    block1 = DataSetBlock(
        data[0:2, :, :],
        aux_data=aux_data,
        slicing_dim=0,
        block_start=0,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        global_shape=GLOBAL_SHAPE,
    )

    block2 = DataSetBlock(
        data[2:4, :, :],
        aux_data=aux_data,
        slicing_dim=0,
        block_start=2,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        global_shape=GLOBAL_SHAPE,
    )

    # darks/flats setters on aux data object should be called when processing first block
    assert wrp._flats_darks_processed is False
    assert darks_setter_spy.call_count == 0
    assert flats_setter_spy.call_count == 0
    newblock1 = wrp.execute(block1)
    assert wrp._flats_darks_processed is True
    # it should only update the darks/flats once, but still apply to the data
    assert darks_setter_spy.call_count == 1
    assert flats_setter_spy.call_count == 1

    # we double them all, so we expect all 3 to be twice the input
    np.testing.assert_array_equal(newblock1.data, 2 * data[0:2, :, :])
    np.testing.assert_array_equal(newblock1.flats, 2 * flats)
    np.testing.assert_array_equal(newblock1.darks, 2 * darks)

    newblock2 = wrp.execute(block2)
    np.testing.assert_array_equal(newblock2.data, 2 * data[2:4, :, :])
    np.testing.assert_array_equal(newblock2.flats, 2 * flats)
    np.testing.assert_array_equal(newblock2.darks, 2 * darks)
    # it should only update the darks/flats once, but still apply to the data
    assert darks_setter_spy.call_count == 1
    assert flats_setter_spy.call_count == 1

    # the flats, darks in original dataset should have changed by reference
    aux_flats = aux_data.get_flats()
    aux_darks = aux_data.get_darks()
    assert aux_flats is not None
    assert aux_darks is not None
    np.testing.assert_array_equal(aux_flats, 2 * flats)
    np.testing.assert_array_equal(aux_darks, 2 * darks)
