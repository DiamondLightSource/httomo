import numpy as np
import pytest
from mpi4py import MPI
from httomo.data.mpiutil import alltoall, alltoall_ring


@pytest.mark.mpi
def test_all_to_all():
    """Original test for backward compatibility - tests the list-returning alltoall."""
    comm = MPI.COMM_WORLD
    data = [np.ones((5, 5, 5), dtype=np.uint16) * comm.rank for _ in range(comm.size)]
    rec = alltoall(data, comm)
    expected = [np.ones((5, 5, 5), dtype=np.uint16) * r for r in range(comm.size)]
    np.testing.assert_array_equal(expected, rec)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "concat_axis,expected_shape_fn",
    [
        (0, lambda size: (5 * size, 5, 5)),
        (1, lambda size: (5, 5 * size, 5)),
        (2, lambda size: (5, 5, 5 * size)),
    ],
)
def test_all_to_all_ring(concat_axis, expected_shape_fn):
    """Test alltoall_ring returns a concatenated array along different axes."""
    comm = MPI.COMM_WORLD

    # Create data: each rank creates arrays filled with its rank number
    data = [np.ones((5, 5, 5), dtype=np.uint16) * comm.rank for _ in range(comm.size)]

    # Call alltoall_ring with specified concat_axis
    result = alltoall_ring(data, comm, concat_axis=concat_axis)

    # Expected shape depends on concat_axis
    expected_shape = expected_shape_fn(comm.size)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {result.shape}"

    # Verify each section came from the correct rank
    for r in range(comm.size):
        start = r * 5
        end = (r + 1) * 5
        expected_slice = np.ones((5, 5, 5), dtype=np.uint16) * r

        # Create the appropriate slice based on concat_axis
        if concat_axis == 0:
            actual_slice = result[start:end, :, :]
        elif concat_axis == 1:
            actual_slice = result[:, start:end, :]
        else:  # concat_axis == 2
            actual_slice = result[:, :, start:end]

        np.testing.assert_array_equal(
            actual_slice,
            expected_slice,
            err_msg=f"Data from rank {r} not found at expected position for concat_axis={concat_axis}",
        )


@pytest.mark.mpi
def test_all_to_all_ring_unequal_sizes():
    """Test alltoall_ring with arrays of different sizes per rank along concat axis."""
    comm = MPI.COMM_WORLD

    # Each rank creates arrays with varying sizes to send to each other rank
    data = []
    for _ in range(comm.size):
        # All ranks send the same size to a given target, but the size varies by sender
        # Rank 0 creates arrays of size 5, rank 1 creates arrays of size 6, etc.
        size = 5 + comm.rank  # Size based on current rank (sender)
        arr = np.ones((size, 4, 3), dtype=np.float32) * comm.rank
        data.append(arr)

    # Call alltoall_ring with concat_axis=0
    result = alltoall_ring(data, comm, concat_axis=0)

    # Expected total size along axis 0: sum of sizes from all ranks
    # Rank 0 contributes 5, rank 1 contributes 6, etc.
    expected_size_axis0 = sum(5 + r for r in range(comm.size))
    expected_shape = (expected_size_axis0, 4, 3)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {result.shape}"

    # Verify each section
    offset = 0
    for r in range(comm.size):
        size = 5 + r  # Size from rank r
        expected_slice = np.ones((size, 4, 3), dtype=np.float32) * r
        np.testing.assert_array_equal(
            result[offset : offset + size, :, :],
            expected_slice,
            err_msg=f"Slice [{offset}:{offset + size}, :, :] should contain data from rank {r}",
        )
        offset += size


@pytest.mark.mpi
def test_all_to_all_ring_single_process():
    """Test alltoall_ring with a single process."""
    comm = MPI.COMM_SELF  # Single process communicator

    data = [np.ones((5, 5, 5), dtype=np.uint16) * 42]

    result = alltoall_ring(data, comm, concat_axis=0)

    # Should return the first array as-is
    expected = np.ones((5, 5, 5), dtype=np.uint16) * 42
    np.testing.assert_array_equal(result, expected)


@pytest.mark.mpi
def test_alltoall_ring_vs_alltoall():
    """Test that alltoall_ring produces equivalent results to alltoall + concatenate."""
    comm = MPI.COMM_WORLD

    # Create test data
    data = [
        np.ones((7, 6, 5), dtype=np.float32) * (comm.rank + 1) for _ in range(comm.size)
    ]

    # Method 1: Use alltoall_ring with concat_axis=0
    result_ring = alltoall_ring(data, comm, concat_axis=0)

    # Method 2: Use original alltoall + concatenate along axis 0
    result_list = alltoall(data, comm)
    result_concat = np.concatenate(result_list, axis=0)

    # They should be identical
    np.testing.assert_array_equal(
        result_ring,
        result_concat,
        err_msg="alltoall_ring should produce the same result as alltoall + concatenate",
    )

    # Verify shapes match
    assert result_ring.shape == result_concat.shape

    # Expected shape: (7*comm.size, 6, 5)
    expected_shape = (7 * comm.size, 6, 5)
    assert result_ring.shape == expected_shape
