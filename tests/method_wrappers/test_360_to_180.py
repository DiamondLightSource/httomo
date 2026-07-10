from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.sino360_to_180 import Sino360to180Wrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from ..testing_utils import make_mock_preview_config, make_mock_repo
from httomo_backends.methods_database.query import Pattern

import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_sino_360_to_180(mocker: MockerFixture):
    GLOBAL_SHAPE = (10, 20, 30)
    GLOBAL_SHAPE_MOD = (5, 20, 30)

    class FakeModule:
        def sino_360_to_180_tester(data):
            np.testing.assert_array_equal(data, 1)
            return data

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.sinogram),
        "mocked_module_path.morph",
        "sino_360_to_180_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
    )
    assert isinstance(wrp, Sino360to180Wrapper)

    aux_data = AuxiliaryData(angles=2.0 * np.ones(GLOBAL_SHAPE[0], dtype=np.float32))
    data = np.ones(
        GLOBAL_SHAPE_MOD, dtype=np.float32
    )  # assuming the data already averaged here by factor of 2
    input = DataSetBlock(
        data[:, 0:3, :],
        slicing_dim=1,
        aux_data=aux_data,
        chunk_shape=GLOBAL_SHAPE_MOD,
        global_shape=GLOBAL_SHAPE_MOD,
    )

    wrp.execute(input)

    assert aux_data.get_angles().shape[0] == 5
