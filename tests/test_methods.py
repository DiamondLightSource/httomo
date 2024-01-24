from httomo.methods import calculate_stats

import numpy as np
import pytest
from mpi4py import MPI


def test_calculate_stats_simple():
    data = np.arange(30, dtype=np.float32).reshape((2, 3, 5)) - 10.0
    ret = calculate_stats(data)

    assert ret == (-10.0, 19.0, np.sum(data), 30)


def test_calculate_stats_with_nan_and_inf():
    data = np.arange(30, dtype=np.float32).reshape((2, 3, 5)) - 10.0
    expected =  np.sum(data) - data[1,1,1] - data[0,2,3] - data[1,2,3]
    data[1,1,1] = float('inf')
    data[0,2,3] = float('-inf')
    data[1,2,3] = float('nan')
    ret = calculate_stats(data)

    assert ret == (-10.0, 19.0, expected, 30)




