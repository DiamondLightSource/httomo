import numpy as np

from httomo.recon.rotation import find_center_360
from numpy.testing import assert_allclose


def test_find_center_360():
    # test a random matrix first
    mat = np.random.rand(100, 100, 100)
    (cor, overlap, side, overlap_position) = find_center_360(mat[:, 2, :])

    eps = 1e-5
    assert_allclose(cor, 76.5484790802002, rtol=eps)
    assert_allclose(overlap, 44.90304183959961, rtol=eps)
    assert side == 1
    assert_allclose(overlap_position, 60.09695816040039, rtol=eps)
