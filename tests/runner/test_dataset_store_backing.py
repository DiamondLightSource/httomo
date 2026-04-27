from typing import List, Tuple

import pytest
import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.runner.dataset_store_backing import (
    DataSetStoreBacking,
    calculate_section_input_chunk_shape,
    calculate_section_output_chunk_shape,
    determine_store_backing,
)
from httomo.runner.output_ref import OutputRef
from httomo.runner.pipeline import Pipeline
from httomo.runner.section import sectionize
from httomo.utils import make_3d_shape_from_shape
from tests.testing_utils import make_test_loader, make_test_method

from httomo_backends.methods_database.query import GpuMemoryRequirement, Pattern


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
    section_chunk_shape = calculate_section_input_chunk_shape(
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
    section_output_chunk_shape = calculate_section_output_chunk_shape(
        chunk_shape=SECTION_INPUT_CHUNK_SHAPE,
        section=sections[0],
    )
    section_output_chunk_bytes = (
        np.prod(section_output_chunk_shape) * np.dtype(DTYPE).itemsize
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
    section_output_chunk_shape = calculate_section_output_chunk_shape(
        chunk_shape=SECTION_INPUT_CHUNK_SHAPE,
        section=sections[0],
    )
    section_output_chunk_bytes = (
        np.prod(section_output_chunk_shape) * np.dtype(DTYPE).itemsize
    )
    assert section_output_chunk_bytes == EXPECTED_SECTION_OUTPUT_CHUNK_BYTES


@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (6 * 1024**2, DataSetStoreBacking.File),
        (7 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["6MB-limit-file-backing", "7MB-limit-ram-backing"],
)
def test_determine_store_backing_reslice_single_proc(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For a single process, chunk shape = global shape
    #
    # The dtype and shape combined makes:
    # - the write chunk ~3.4MB
    # - the read chunk also ~3.4MB
    # - reslice shouldn't occur due to running with a single process
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(mocker=mocker, method_name="m1", pattern=Pattern.projection)
    m2 = make_test_method(mocker=mocker, method_name="m2", pattern=Pattern.sinogram)

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing


@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (6 * 1024**2, DataSetStoreBacking.File),
        (7 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["6MB-limit-file-backing", "7MB-limit-ram-backing"],
)
def test_determine_store_backing_no_reslice_single_proc(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For a single process, chunk shape = global shape
    #
    # The dtype and shape combined makes:
    # - the write chunk ~3.4MB
    # - the read chunk also ~3.4MB
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(mocker=mocker, method_name="m1", pattern=Pattern.projection)
    m2 = make_test_method(
        mocker=mocker,
        method_name="m2",
        pattern=Pattern.projection,
        some_param=(OutputRef(m1, "some-param")),
    )

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (6 * 1024**2, DataSetStoreBacking.File),
        (7 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["6MB-limit-file-backing", "7MB-limit-ram-backing"],
)
def test_determine_store_backing_reslice_two_procs(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For two processes, chunk shape = half of global shape
    #
    # The dtype and shape combined makes:
    # - the write chunk ~1.7MB
    # - the read chunk also ~1.7MB
    # - the output of reslice also ~1.7MB
    # - the intermediate data created by reslice algorithm ~1.7MB
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(mocker=mocker, method_name="m1", pattern=Pattern.projection)
    m2 = make_test_method(mocker=mocker, method_name="m2", pattern=Pattern.sinogram)

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (3 * 1024**2, DataSetStoreBacking.File),
        (4 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["3MB-limit-file-backing", "4MB-limit-ram-backing"],
)
def test_determine_store_backing_no_reslice_two_procs(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For two processes, chunk shape = half of global shape
    #
    # The dtype and shape combined makes:
    # - the write chunk ~1.7MB
    # - the read chunk also ~1.7MB
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(mocker=mocker, method_name="m1", pattern=Pattern.projection)
    m2 = make_test_method(
        mocker=mocker,
        method_name="m2",
        pattern=Pattern.projection,
        some_param=(OutputRef(m1, "some-param")),
    )

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing


@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (41 * 1024**2, DataSetStoreBacking.File),
        (42 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["41MB-limit-file-backing", "42MB-limit-ram-backing"],
)
def test_determine_store_backing_large_padding_reslice_single_proc(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For a single process, chunk shape = global shape
    #
    # The dtype and shape combined makes:
    # - the write chunk ~3.4MB
    # - the padded input chunk ~37.7MB (110 * 300 * 300 * 4 / (1024 ** 2))
    # - reslice shouldn't occur due to running with a single process
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)
    PADDING = (50, 50)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(
        mocker=mocker, method_name="m1", pattern=Pattern.projection, padding=True
    )
    m2 = make_test_method(mocker=mocker, method_name="m2", pattern=Pattern.sinogram)
    mocker.patch.object(
        target=m1,
        attribute="calculate_padding",
        return_value=PADDING,
    )

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing


@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (41 * 1024**2, DataSetStoreBacking.File),
        (42 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["41MB-limit-file-backing", "42MB-limit-ram-backing"],
)
def test_determine_store_backing_large_padding_no_reslice_single_proc(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For a single process, chunk shape = global shape
    #
    # The dtype, shape, and padding combined makes:
    # - the unpadded input chunk ~3.4MB (10 * 300 * 300 * 4 / (1024 ** 2))
    # - the padded input chunk ~37.7MB (110 * 300 * 300 * 4 / (1024 ** 2))
    # - the output chunk ~3.4MB (10 * 300 * 300 * 4 / (1024 ** 2))
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)
    PADDING = (50, 50)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(
        mocker=mocker, method_name="m1", pattern=Pattern.projection, padding=True
    )
    m2 = make_test_method(
        mocker=mocker,
        method_name="m2",
        pattern=Pattern.projection,
        some_param=(OutputRef(m1, "some-param")),
    )
    mocker.patch.object(
        target=m1,
        attribute="calculate_padding",
        return_value=PADDING,
    )

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)
    print(sections)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (41 * 1024**2, DataSetStoreBacking.File),
        (42 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["41MB-limit-file-backing", "42MB-limit-ram-backing"],
)
def test_determine_store_backing_large_padding_reslice_two_procs(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For two processes, chunk shape = half of global shape
    #
    # The dtype and shape combined makes:
    # - the write chunk ~1.7MB
    # - the padded input chunk ~36.0MB (105 * 300 * 300 * 4 / (1024 ** 2))
    # - the output of reslice also ~1.7MB
    # - the intermediate data created by reslice algorithm ~1.7MB
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)
    PADDING = (50, 50)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(
        mocker=mocker, method_name="m1", pattern=Pattern.projection, padding=True
    )
    m2 = make_test_method(mocker=mocker, method_name="m2", pattern=Pattern.sinogram)
    mocker.patch.object(
        target=m1,
        attribute="calculate_padding",
        return_value=PADDING,
    )

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
@pytest.mark.parametrize(
    "memory_limit, expected_store_backing",
    [
        (37 * 1024**2, DataSetStoreBacking.File),
        (38 * 1024**2, DataSetStoreBacking.RAM),
    ],
    ids=["37MB-limit-file-backing", "38MB-limit-ram-backing"],
)
def test_determine_store_backing_large_padding_no_reslice_two_procs(
    mocker: MockerFixture,
    memory_limit: int,
    expected_store_backing: DataSetStoreBacking,
):
    COMM = MPI.COMM_WORLD

    # For a single process, chunk shape = global shape
    #
    # The dtype, shape, and padding combined makes:
    # - the unpadded input chunk ~1.7MB (5 * 300 * 300 * 4 / (1024 ** 2))
    # - the padded input chunk ~36.0MB (105 * 300 * 300 * 4 / (1024 ** 2))
    # - the output chunk ~1.7MB (5 * 300 * 300 * 4 / (1024 ** 2))
    DTYPE = np.float32
    GLOBAL_SHAPE = (10, 300, 300)
    PADDING = (50, 50)

    # Define dummy loader and method wrapper objects
    loader = make_test_loader(mocker=mocker)
    m1 = make_test_method(
        mocker=mocker, method_name="m1", pattern=Pattern.projection, padding=True
    )
    m2 = make_test_method(
        mocker=mocker,
        method_name="m2",
        pattern=Pattern.projection,
        some_param=(OutputRef(m1, "some-param")),
    )
    mocker.patch.object(
        target=m1,
        attribute="calculate_padding",
        return_value=PADDING,
    )

    # Get list of section objects that represent pipeline
    pipeline = Pipeline(
        loader=loader,
        methods=[m1, m2],
    )
    sections = sectionize(pipeline)

    store_backing = determine_store_backing(
        comm=COMM,
        sections=sections,
        memory_limit_bytes=memory_limit,
        dtype=DTYPE,
        global_shape=GLOBAL_SHAPE,
        section_idx=0,
    )
    assert store_backing is expected_store_backing
