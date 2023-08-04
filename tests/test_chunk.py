from typing import List, Tuple

import pytest
import numpy as np
from unittest import mock

from httomo.data.hdf._utils.chunk import get_data_bounds


@pytest.mark.parametrize(
    "nproc, data, allgather_return, expected_bounds",
    [
        # Case
        # - 1 MPI process / serial run
        # - remove no projections
        (
            1,
            [np.empty((180, 128, 160))],
            [(180, 128, 160)],
            [(0, 180)],
        ),
        # Case
        # - 2 MPI processes
        # - remove no projections from either process
        (
            2,
            [np.empty((90, 128, 160)), np.empty((90, 128, 160))],
            [(90, 128, 160), (90, 128, 160)],
            [(0, 90), (90, 180)],
        ),
        # Case
        # - 2 MPI processes
        # - remove 5 projections belonging to rank 0 process
        (
            2,
            [np.empty((85, 128, 160)), np.empty((90, 128, 160))],
            [(85, 128, 160), (90, 128, 160)],
            [(0, 85), (85, 175)],
        ),
        # Case
        # - 2 MPI processes
        # - remove 5 projections belonging to rank 1 process
        (
            2,
            [np.empty((90, 128, 160)), np.empty((85, 128, 160))],
            [(90, 128, 160), (85, 128, 160)],
            [(0, 90), (90, 175)],
        ),
        # Case
        # - 4 MPI processes
        # - remove no projections
        (
            4,
            [
                np.empty((45, 128, 160)),
                np.empty((45, 128, 160)),
                np.empty((45, 128, 160)),
                np.empty((45, 128, 160)),
            ],
            [(45, 128, 160), (45, 128, 160), (45, 128, 160), (45, 128, 160)],
            [(0, 45), (45, 90), (90, 135), (135, 180)],
        ),
        # Case
        # - 4 MPI processes
        # - remove 5 projections from rank 0 process
        (
            4,
            [
                np.empty((40, 128, 160)),
                np.empty((45, 128, 160)),
                np.empty((45, 128, 160)),
                np.empty((45, 128, 160)),
            ],
            [(40, 128, 160), (45, 128, 160), (45, 128, 160), (45, 128, 160)],
            [(0, 40), (40, 85), (85, 130), (130, 175)],
        ),
        # Case
        # - 4 MPI processes
        # - remove 5 projections from rank 2 process
        (
            4,
            [
                np.empty((45, 128, 160)),
                np.empty((45, 128, 160)),
                np.empty((40, 128, 160)),
                np.empty((45, 128, 160)),
            ],
            [(45, 128, 160), (45, 128, 160), (40, 128, 160), (45, 128, 160)],
            [(0, 45), (45, 90), (90, 130), (130, 175)],
        ),
        # Case
        # - 4 MPI processes
        # - remove 5 projections from rank 1 process
        # - remove 5 projections from rank 3 process
        (
            4,
            [
                np.empty((45, 128, 160)),
                np.empty((40, 128, 160)),
                np.empty((45, 128, 160)),
                np.empty((40, 128, 160)),
            ],
            [(45, 128, 160), (40, 128, 160), (45, 128, 160), (40, 128, 160)],
            [(0, 45), (45, 85), (85, 130), (130, 170)],
        ),
    ],
    # ID naming:
    # {NO_OF_PROCS}-proc-{SLICE_DIM_LEN}-{PROC_0_LEN}-{PROC_1_LEN}-...
    ids=[
        "1-proc-180-180",
        "2-proc-180-90-90",
        "2-proc-180-85-90",
        "2-proc-180-90-85",
        "4-proc-180-45-45-45-45",
        "4-proc-180-40-45-45-45",
        "4-proc-180-45-45-40-45",
        "4-proc-180-45-40-45-40",
    ],
)
def test_get_data_bounds_projs(
    nproc: int,
    data: List[np.ndarray],
    allgather_return: List[Tuple[int, int, int]],
    expected_bounds: List[Tuple[int, int]],
) -> None:
    # Mock MPI communicator object
    comm = mock.Mock()
    # Mock the return of the `.allgather()` method on a communicator object
    comm.allgather.return_value = allgather_return

    # Fake each MPI process calling `get_data_bounds()` by running the function
    # with a different data and rank
    returned_bounds = []
    for i in range(nproc):
        comm.rank = i
        bounds = get_data_bounds(data[i], comm, 0)
        returned_bounds.append(bounds)

    # Check that the bounds for each "process" matches the expected values
    for i in range(nproc):
        assert returned_bounds[i][0] == expected_bounds[i][0]
        assert returned_bounds[i][1] == expected_bounds[i][1]
