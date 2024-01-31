from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.reconstruction import ReconstructionWrapper
from httomo.runner.dataset import DataSet
from ..testing_utils import make_mock_repo


import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_recon_handles_reconstruction_angle_reshape(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        # we give the angles a different name on purpose
        def recon_tester(data, theta):
            np.testing.assert_array_equal(data, 1)
            np.testing.assert_array_equal(theta, 2)
            assert data.shape[0] == len(theta)
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
    )
    assert isinstance(wrp, ReconstructionWrapper)
    dummy_dataset.data[:] = 1
    dummy_dataset.unlock()
    dummy_dataset.angles[:] = 2
    dummy_dataset.lock()

    input = dummy_dataset.make_block(0, 0, 3)
    wrp.execute(input)


def test_recon_handles_reconstruction_axisswap(
    mocker: MockerFixture, dummy_dataset: DataSet
):
    class FakeModule:
        def recon_tester(data, theta):
            return data.swapaxes(0, 1)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, swap_dims_on_output=True),
        "mocked_module_path.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
    )
    dummy_dataset = DataSet(
        data=np.ones((13, 14, 15), dtype=np.float32),
        angles=dummy_dataset.angles,
        flats=dummy_dataset.flats,
        darks=dummy_dataset.darks,
    )
    res = wrp.execute(dummy_dataset.make_block(0))

    assert res.data.shape == (13, 14, 15)

def test_recon_changes_global_shape_if_size_changes(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def recon_tester(data, theta):
            data = np.ones((30, data.shape[1], 15), dtype=np.float32)
            return data.swapaxes(0, 1)

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, swap_dims_on_output=True),
        "mocked_module_path.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
    )
    dummy_dataset = DataSet(
        data=np.ones((13, 14, 15), dtype=np.float32),
        angles=dummy_dataset.angles,
        flats=dummy_dataset.flats,
        darks=dummy_dataset.darks,
    )
    res = wrp.execute(dummy_dataset.make_block(1, 0, 3))
    
    assert res.shape == (30, 3, 15)
    assert res.global_shape == (30, 14, 15)


def test_recon_gives_recon_algorithm(mocker: MockerFixture):
    class FakeModule:
        def recon_tester(data, algorithm, center):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "test.algorithm",
        "recon_tester",
        MPI.COMM_WORLD,
    )

    assert wrp.recon_algorithm is None
    wrp["algorithm"] = "testalgo"
    assert wrp.recon_algorithm == "testalgo"


def test_recon_gives_no_recon_algorithm_if_not_recon_method(mocker: MockerFixture):
    class FakeModule:
        def tester(data, algorithm):
            return data

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "test.something",
        "tester",
        MPI.COMM_WORLD,
    )

    assert wrp.recon_algorithm is None
    wrp["algorithm"] = "testalgo"
    assert wrp.recon_algorithm is None