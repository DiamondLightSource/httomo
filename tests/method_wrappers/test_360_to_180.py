from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.average_frames import AverageFramesWrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from ..testing_utils import make_mock_preview_config, make_mock_repo
from httomo_backends.methods_database.query import Pattern

import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture

def test_angle_averaging(mocker: MockerFixture):
    GLOBAL_SHAPE = (10, 20, 30)

    class FakeModule:
        def average_projection_frames_tester(data, projection_averaging_factor):
            np.testing.assert_array_equal(data, 1)
            return data

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, pattern=Pattern.sinogram),
        "mocked_module_path.morph",
        "average_projection_frames_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        projection_averaging_factor = 2,
    )
    assert isinstance(wrp, AverageFramesWrapper)

    aux_data = AuxiliaryData(
        angles=2.0 * np.ones(GLOBAL_SHAPE[0] + 10, dtype=np.float32)
    )
    data = np.ones(GLOBAL_SHAPE, dtype=np.float32) # assuming the data already averaged here by factor of 2
    input = DataSetBlock(
        data[:, 0:3, :],
        slicing_dim=1,
        aux_data=aux_data,
        chunk_shape=GLOBAL_SHAPE,
        global_shape=GLOBAL_SHAPE,
    )

    wrp.execute(input)

    assert aux_data.get_angles().shape[0] == 5