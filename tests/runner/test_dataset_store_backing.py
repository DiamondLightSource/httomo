from typing import List, Tuple

import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.runner.dataset_store_backing import calculate_section_chunk_shape
from httomo.utils import make_3d_shape_from_shape


@pytest.mark.parametrize(
    "nprocs, rank, section_slicing_dim, section_padding",
    [
        (2, 1, 0, (0, 0)),
        (2, 1, 0, (3, 5)),
        (2, 1, 1, (0, 0)),
        (2, 1, 1, (3, 5)),
        (4, 2, 0, (0, 0)),
        (4, 2, 0, (3, 5)),
        (4, 2, 1, (0, 0)),
        (4, 2, 1, (3, 5)),
    ],
    ids=[
        "2procs-proj-to-proj_unpadded",
        "2procs-proj-to-proj_padded",
        "2procs-proj-to-sino_unpadded",
        "2procs-proj-to-sino_padded",
        "4procs-proj-to-proj_unpadded",
        "4procs-proj-to-proj_padded",
        "4procs-proj-to-sino_unpadded",
        "4procs-proj-to-sino_padded",
    ],
)
def test_calculate_section_chunk_shape(
    nprocs: int,
    rank: int,
    section_slicing_dim: int,
    section_padding: Tuple[int, int],
    mocker: MockerFixture,
):
    GLOBAL_SHAPE = (1801, 2160, 2560)

    # Define mock communicator that reflects the desired data splitting/distribution to be
    # tested
    mock_global_comm = mocker.create_autospec(spec=MPI.Comm, size=nprocs, rank=rank)

    # The chunk shape for the section should reflect the padding needed for that section
    expected_chunk_shape: List[int] = list(GLOBAL_SHAPE)
    start = round(GLOBAL_SHAPE[section_slicing_dim] / nprocs * rank)
    stop = round(GLOBAL_SHAPE[section_slicing_dim] / nprocs * (rank + 1))
    slicing_dim_len = stop - start
    expected_chunk_shape[section_slicing_dim] = (
        slicing_dim_len + section_padding[0] + section_padding[1]
    )
    section_chunk_shape = calculate_section_chunk_shape(
        comm=mock_global_comm,
        global_shape=GLOBAL_SHAPE,
        slicing_dim=section_slicing_dim,
        padding=section_padding,
    )
    assert section_chunk_shape == make_3d_shape_from_shape(expected_chunk_shape)
