from httomo.method_wrappers import make_method_wrapper
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from ..testing_utils import make_mock_preview_config, make_mock_repo

import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_recon_handles_reconstruction_axisswap(mocker: MockerFixture):
    class FakeModule:
        def recon_tester(data, theta):
            return data.swapaxes(0, 1)

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, swap_dims_on_output=True),
        "mocked_module_path.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
    )
    block = DataSetBlock(
        data=np.ones((13, 14, 15), dtype=np.float32),
        aux_data=AuxiliaryData(angles=np.ones(13, dtype=np.float32)),
        slicing_dim=1,
    )
    res = wrp.execute(block)

    assert res.data.shape == (13, 14, 15)


def test_recon_changes_global_shape_if_size_changes(mocker: MockerFixture):
    class FakeModule:
        def recon_tester(data, theta):
            data = np.ones((30, data.shape[1], 15), dtype=np.float32)
            return data.swapaxes(0, 1)

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, swap_dims_on_output=True),
        "mocked_module_path.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
    )
    block = DataSetBlock(
        data=np.ones((13, 3, 15), dtype=np.float32),
        aux_data=AuxiliaryData(angles=np.ones(13, dtype=np.float32)),
        slicing_dim=1,
        global_shape=(13, 14, 15),
        chunk_shape=(13, 14, 15),
    )
    res = wrp.execute(block)

    assert res.shape == (30, 3, 15)
    assert res.global_shape == (30, 14, 15)
    assert res.chunk_shape == (30, 14, 15)


def test_recon_gives_recon_algorithm(mocker: MockerFixture):
    class FakeModule:
        def recon_tester(data, algorithm, center):
            return data

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "test.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
    )

    assert wrp.recon_algorithm is None
    wrp["algorithm"] = "testalgo"
    assert wrp.recon_algorithm == "testalgo"


def test_recon_gives_no_recon_algorithm_if_not_recon_method(mocker: MockerFixture):
    class FakeModule:
        def tester(data, algorithm):
            return data

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "test.something",
        "tester",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
    )

    assert wrp.recon_algorithm is None
    wrp["algorithm"] = "testalgo"
    assert wrp.recon_algorithm is None
