import math
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.datareducer import DatareducerWrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from ..testing_utils import make_mock_repo


import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_datareducer(mocker: MockerFixture):
    class FakeModule:
        def data_reducer(x):
            return 2 * x

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.morph",
        "data_reducer",
        MPI.COMM_WORLD,
    )
    assert isinstance(wrp, DatareducerWrapper)

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
    block1 = wrp.execute(block1)
    assert wrp._flats_darks_processed is True
    assert darks_setter_spy.call_count == 1
    assert flats_setter_spy.call_count == 1

    # we double flats and darks only, so we expect them to be twice the original
    np.testing.assert_array_equal(block1.data, data[0:2, :, :])
    np.testing.assert_array_equal(block1.flats, 2 * flats)
    np.testing.assert_array_equal(block1.darks, 2 * darks)

    # darks/flats setters on aux data object should NOT be called when processing second block
    block2 = wrp.execute(block2)
    assert darks_setter_spy.call_count == 1
    assert flats_setter_spy.call_count == 1

    # darks/flats should NOT have been doubled a second time when the second block was
    # processed (ie, they should still be only double the original, not quadruple the original)
    np.testing.assert_array_equal(block1.data, data[0:2, :, :])
    np.testing.assert_array_equal(block1.flats, 2 * flats)
    np.testing.assert_array_equal(block1.darks, 2 * darks)
    np.testing.assert_array_equal(block2.data, data[2:4, :, :])
    np.testing.assert_array_equal(block2.flats, 2 * flats)
    np.testing.assert_array_equal(block2.darks, 2 * darks)
