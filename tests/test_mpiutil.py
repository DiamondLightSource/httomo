import numpy as np
import pytest
from mpi4py import MPI
from httomo.data.mpiutil import alltoall, alltoall_ring

@pytest.mark.mpi
def test_all_to_all_ring():
    """Test alltoall_ring returns a concatenated array instead of a list."""
    comm = MPI.COMM_WORLD
    
    # Create data: each rank creates arrays filled with its rank number
    data = [np.ones((5, 5, 5), dtype=np.uint16) * comm.rank for _ in range(comm.size)]
    
    # Call alltoall_ring with concat_axis=2 (concatenate along third dimension)
    result = alltoall_ring(data, comm, concat_axis=2)
    
    # Expected: concatenated array with shape (5, 5, 5*comm.size)
    # Each slice of 5 along axis 2 should be filled with the rank number it came from
    expected_shape = (5, 5, 5 * comm.size)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    
    # Verify each section came from the correct rank
    for r in range(comm.size):
        start = r * 5
        end = (r + 1) * 5
        expected_slice = np.ones((5, 5, 5), dtype=np.uint16) * r
        np.testing.assert_array_equal(
            result[:, :, start:end], 
            expected_slice,
            err_msg=f"Slice [:, :, {start}:{end}] should contain data from rank {r}"
        )

@pytest.mark.mpi
def test_all_to_all_ring_axis1():
    """Test alltoall_ring with concatenation along axis 1 (middle dimension)."""
    comm = MPI.COMM_WORLD
    
    # Create data with different values per rank
    data = [np.ones((5, 5, 5), dtype=np.float32) * (comm.rank + 1) for _ in range(comm.size)]
    
    # Call alltoall_ring with concat_axis=1
    result = alltoall_ring(data, comm, concat_axis=1)
    
    # Expected shape: (5, 5*comm.size, 5)
    expected_shape = (5, 5 * comm.size, 5)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    
    # Verify each section along axis 1
    for r in range(comm.size):
        start = r * 5
        end = (r + 1) * 5
        expected_slice = np.ones((5, 5, 5), dtype=np.float32) * (r + 1)
        np.testing.assert_array_equal(
            result[:, start:end, :],
            expected_slice,
            err_msg=f"Slice [:, {start}:{end}, :] should contain data from rank {r}"
        )

@pytest.mark.mpi
def test_all_to_all_ring_unequal_sizes():
    """Test alltoall_ring with arrays of different sizes per rank along concat axis."""
    comm = MPI.COMM_WORLD
    
    # Create arrays with varying sizes along the concat axis (axis 2)
    # Each rank sends different sized chunks to other ranks
    data = []
    for target_rank in range(comm.size):
        # Size varies: rank 0 sends size 5, rank 1 sends size 6, etc.
        size = 5 + target_rank
        arr = np.ones((4, 3, size), dtype=np.float32) * comm.rank
        data.append(arr)
    
    # Call alltoall_ring with concat_axis=2
    result = alltoall_ring(data, comm, concat_axis=2)
    
    # Expected total size along axis 2: sum of all sizes
    expected_size_axis2 = sum(5 + r for r in range(comm.size))
    expected_shape = (4, 3, expected_size_axis2)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    
    # Verify each section
    offset = 0
    for r in range(comm.size):
        size = 5 + r
        expected_slice = np.ones((4, 3, size), dtype=np.float32) * r
        np.testing.assert_array_equal(
            result[:, :, offset:offset + size],
            expected_slice,
            err_msg=f"Slice [:, :, {offset}:{offset + size}] should contain data from rank {r}"
        )
        offset += size

@pytest.mark.mpi
def test_all_to_all_ring_single_process():
    """Test alltoall_ring with a single process."""
    comm = MPI.COMM_SELF  # Single process communicator
    
    data = [np.ones((5, 5, 5), dtype=np.uint16) * 42]
    
    result = alltoall_ring(data, comm, concat_axis=2)
    
    # Should return the first array as-is
    expected = np.ones((5, 5, 5), dtype=np.uint16) * 42
    np.testing.assert_array_equal(result, expected)
