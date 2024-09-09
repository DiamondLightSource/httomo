from typing import List, Tuple

import pytest
import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.runner.dataset_store_backing import (
    calculate_section_chunk_shape,
    calculate_section_chunk_bytes,
)
from httomo.runner.methods_repository_interface import GpuMemoryRequirement
from httomo.runner.pipeline import Pipeline
from httomo.runner.section import sectionize
from httomo.utils import Pattern, make_3d_shape_from_shape
from tests.testing_utils import make_test_loader, make_test_method


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


def test_calculate_section_chunk_bytes_output_dims_change(mocker: MockerFixture):
    NEW_OUTPUT_DIMS = (300, 400)
    SECTION_INPUT_CHUNK_SHAPE = (100, 10, 100)
    DTYPE = np.float32
    EXPECTED_SECTION_OUTPUT_CHUNK_SHAPE = (
        SECTION_INPUT_CHUNK_SHAPE[0],
        NEW_OUTPUT_DIMS[0],
        NEW_OUTPUT_DIMS[1],
    )
    EXPECTED_SECTION_OUTPUT_CHUNK_BYTES = (
        np.prod(EXPECTED_SECTION_OUTPUT_CHUNK_SHAPE) * np.dtype(DTYPE).itemsize
    )

    # Define methods to form section, one of which changes the output shape
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(mocker=mocker, method_name="m1", pattern=Pattern.projection)
    m2 = make_test_method(
        mocker=mocker, method_name="m2", pattern=Pattern.projection, gpu=True
    )
    mocker.patch.object(
        m2,
        "memory_gpu",
        [GpuMemoryRequirement(multiplier=2.0, method="direct")],
    )
    mocker.patch.object(
        target=m2,
        attribute="calculate_output_dims",
        return_value=NEW_OUTPUT_DIMS,
    )

    # Generate list of sections
    pipeline = Pipeline(loader=loader, methods=[m1, m2])
    sections = sectionize(pipeline)

    # Check that the number of bytes in the chunk accounts for the non-slicing dims change by
    # the method in the section
    section_output_chunk_bytes = calculate_section_chunk_bytes(
        chunk_shape=SECTION_INPUT_CHUNK_SHAPE,
        dtype=DTYPE,
        section=sections[0],
    )
    assert section_output_chunk_bytes == EXPECTED_SECTION_OUTPUT_CHUNK_BYTES


def test_calculate_section_chunk_bytes_output_dims_change_and_swap(
    mocker: MockerFixture,
):
    RECON_SIZE = 400
    SECTION_INPUT_CHUNK_SHAPE = (100, 10, 100)
    DTYPE = np.float32
    EXPECTED_SECTION_OUTPUT_CHUNK_SHAPE = (
        SECTION_INPUT_CHUNK_SHAPE[1],
        RECON_SIZE,
        RECON_SIZE,
    )
    EXPECTED_SECTION_OUTPUT_CHUNK_BYTES = (
        np.prod(EXPECTED_SECTION_OUTPUT_CHUNK_SHAPE) * np.dtype(DTYPE).itemsize
    )

    # Define methods to form section, one of which changes the output shape and also swaps the
    # output dims 0 and 1 (which can happen in recon methods)
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(
        mocker=mocker, method_name="stripe-removal", pattern=Pattern.sinogram
    )
    # NOTE: The reason that `swap_dims_on_output=True` is not passed to `make_test_method()`
    # even though `m2` is assumed to be swapping the output dims is because that doesn't
    # actually do anything when not running the method on data.
    #
    # The output dims being swapped is not taken into account when calling
    # `calculate_output_dims()` on a method wrapper object, and the output dims swapping is
    # implicitly done when the method executes.
    #
    # It however still seemed reasonable to check that the correct chunk bytes value was given
    # if there's a method in a section that swaps the output dims.
    m2 = make_test_method(
        mocker=mocker,
        method_name="recon",
        pattern=Pattern.sinogram,
        gpu=True,
    )
    mocker.patch.object(
        m2,
        "memory_gpu",
        [GpuMemoryRequirement(multiplier=2.0, method="direct")],
    )
    mocker.patch.object(
        target=m2,
        attribute="calculate_output_dims",
        return_value=(RECON_SIZE, RECON_SIZE),
    )

    # Generate list of sections
    pipeline = Pipeline(loader=loader, methods=[m1, m2])
    sections = sectionize(pipeline)

    # Check that the number of bytes in the chunk accounts for the non-slicing dims change by
    # the method in the section
    section_output_chunk_bytes = calculate_section_chunk_bytes(
        chunk_shape=SECTION_INPUT_CHUNK_SHAPE,
        dtype=DTYPE,
        section=sections[0],
    )
    assert section_output_chunk_bytes == EXPECTED_SECTION_OUTPUT_CHUNK_BYTES
