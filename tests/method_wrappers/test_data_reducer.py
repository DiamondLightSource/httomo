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

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.morph",
        "data_reducer",
        MPI.COMM_WORLD,
    )
    assert isinstance(wrp, DatareducerWrapper)

    GLOBAL_SHAPE = (10, 20, 30)
    aux_data = AuxiliaryData(
        angles=np.linspace(0, math.pi, GLOBAL_SHAPE[0], dtype=np.float32),
        darks=4 * np.ones((2, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), dtype=np.float32),
        flats=3 * np.ones((3, GLOBAL_SHAPE[1], GLOBAL_SHAPE[2]), dtype=np.float32),
    )
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
    newblock1 = wrp.execute(block1)

    # we double flats and darks only, so we expect them to be twice the input
    np.testing.assert_array_equal(newblock1.data, 1)
    np.testing.assert_array_equal(newblock1.flats, 6)
    np.testing.assert_array_equal(newblock1.darks, 8)

    # it should only update the darks/flats once
    block2 = DataSetBlock(
        data[2:4, :, :],
        aux_data=aux_data,
        slicing_dim=0,
        block_start=2,
        chunk_start=0,
        chunk_shape=GLOBAL_SHAPE,
        global_shape=GLOBAL_SHAPE,
    )
    newblock2 = wrp.execute(block2)
    np.testing.assert_array_equal(newblock2.data, 1)
    np.testing.assert_array_equal(newblock2.flats, 6)
    np.testing.assert_array_equal(newblock2.darks, 8)

    # the flats, darks should have changed by reference
    aux_flats = aux_data.get_flats()
    aux_darks = aux_data.get_darks()
    assert aux_flats is not None
    assert aux_darks is not None
    np.testing.assert_array_equal(aux_flats, 6)
    np.testing.assert_array_equal(aux_darks, 8)
