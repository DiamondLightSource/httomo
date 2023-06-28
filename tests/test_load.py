import numpy as np
import pytest
from mpi4py import MPI

from httomo.data.hdf._utils.load import load_data

comm = MPI.COMM_WORLD


@pytest.mark.parametrize(
    "file, dim, path_to_data, expected_shape, expected_sum, expected_mean",
    [
        (
            "tests/test_data/tomo_standard.nxs",
            1,
            "entry1/tomo_entry/data/data",
            (180, 128, 160),
            2982481340,
            809.0498426649306,
        ),
        (
            "tests/test_data/tomo_standard.nxs",
            3,
            "entry1/tomo_entry/data/data",
            (220, 128, 160),
            3382481340,
            750.7282803622159,
        ),
        (
            "tests/test_data/k11_diad/k11-18014.nxs",
            1,
            "/entry/imaging/data",
            (180, 22, 26),
            3057017180,
            29691.309052,
        ),
    ],
    ids=["1d-standard", "3d-standard", "1d-diad"],
)
def test_load_data(
    file, dim, path_to_data, expected_shape, expected_sum, expected_mean
):
    preview = "0:180, :, :"
    output = load_data(file, dim, path_to_data, preview, comm=comm)

    assert output.shape == expected_shape
    assert output.dtype == np.uint16
    assert output.sum() == expected_sum
    np.testing.assert_allclose(output.mean(), expected_mean, rtol=1e-6)
