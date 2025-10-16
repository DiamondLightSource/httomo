from typing import List

import logging

import numpy as np
from mpi4py import MPI

from httomo.utils import log_once


__all__ = ["alltoall", "alltoall_ring"]


# add this here so that we can mock it in the tests
_mpi_max_elements = 2**31


def alltoall_ring(
    arrays: List[np.ndarray], comm: MPI.Comm, concat_axis: int = 0
) -> np.ndarray:
    """Distributes a list of contiguous numpy arrays from each rank to every other rank
    using a ring communication pattern for reduced memory usage.

    This implementation uses point-to-point communication in a ring pattern instead of
    collective Alltoallv, trading some performance for significantly lower memory usage.
    It only keeps one send/receive pair in memory at a time.

    The received arrays are directly written into a pre-allocated concatenated output array,
    eliminating the need for a separate concatenation step and reducing memory usage.

    Parameters
    ----------
    arrays : List[np.ndarray]
        List of 3D numpy arrays to be distributed. Length must be the full size of the given
        communicator. arrays[i] will be sent to rank i.

    comm : MPI.Comm
        MPI communicator

    concat_axis : int
        The axis along which received arrays should be concatenated (default: 0)

    Returns
    -------
    np.ndarray
        A single concatenated array containing all received data along the specified axis.
        The data from rank i is placed at the appropriate offset along concat_axis.
    """
    rank = comm.rank
    nprocs = comm.size

    log_once(
        f"alltoall_ring: Starting with {len(arrays)} arrays, comm.size={nprocs}, concat_axis={concat_axis}",
        level=logging.DEBUG,
    )

    if len(arrays) != comm.size:
        err_str = "list of arrays for MPI alltoall call must match communicator size"
        raise ValueError(err_str)

    assert all(type(a) is np.ndarray for a in arrays), "All arrays must be numpy arrays"
    assert all(
        a.dtype == arrays[0].dtype for a in arrays
    ), "All arrays must be of the same type"
    assert arrays[0].dtype in [
        np.float32,
        np.uint16,
    ], "Only 16bit unsigned ints or single precision floats are implemented"
    assert all(a.ndim == 3 for a in arrays), "Only 3D arrays are supported"

    # no MPI or only one process
    if comm.size == 1:
        log_once(
            "alltoall_ring: Single process, returning first array as-is",
            level=logging.DEBUG,
        )
        return arrays[0]

    shapes_send = [a.shape for a in arrays]

    # Exchange shape information first
    shapes_rec = comm.alltoall(shapes_send)

    # Calculate output shape for concatenated array
    output_shape = list(shapes_rec[0])
    output_shape[concat_axis] = sum(s[concat_axis] for s in shapes_rec)

    # Pre-allocate the final concatenated output array
    output_array = np.empty(output_shape, dtype=arrays[0].dtype)

    # Calculate offsets for each rank's data in the output array
    offsets = [0]
    for i in range(nprocs - 1):
        offsets.append(offsets[-1] + shapes_rec[i][concat_axis])

    # Use ring pattern: each process sends to (rank + offset) and receives from (rank - offset)
    for offset in range(nprocs):
        send_to = (rank + offset) % nprocs
        recv_from = (rank - offset) % nprocs

        # Create slice for where to write received data
        recv_slices = [slice(None)] * 3  # 3 is the number of dimension
        recv_offset = offsets[recv_from]
        recv_size = shapes_rec[recv_from][concat_axis]
        recv_slices[concat_axis] = slice(recv_offset, recv_offset + recv_size)

        if send_to == rank:
            # Self-copy (no communication)
            output_array[tuple(recv_slices)] = arrays[rank]
        else:
            # Send arrays[send_to] TO send_to, and receive FROM recv_from
            send_array = arrays[send_to]
            if not send_array.flags.c_contiguous:
                send_array = np.ascontiguousarray(send_array)

            # Get view into output array for receiving
            # This view might not be contiguous, so we need to handle that
            recv_view = output_array[tuple(recv_slices)]

            # Check if recv_view is contiguous, if not we need a temporary buffer
            use_temp_buffer = not recv_view.flags.c_contiguous

            if use_temp_buffer:
                # Create a temporary contiguous buffer
                temp_recv = np.empty(shapes_rec[recv_from], dtype=arrays[0].dtype)
                actual_recv_buffer = temp_recv
            else:
                actual_recv_buffer = recv_view

            # Chunk if array is too large
            max_elements = _mpi_max_elements - 1

            # Use the LARGER of send/recv size to determine chunking
            # This ensures both sides agree on the number of chunks
            transfer_size = max(send_array.size, actual_recv_buffer.size)

            if transfer_size > max_elements:
                # Send in chunks
                send_flat = send_array.ravel()
                recv_flat = actual_recv_buffer.ravel()

                # Calculate number of chunks based on the larger size
                num_chunks = (transfer_size + max_elements - 1) // max_elements

                for chunk_idx in range(num_chunks):
                    # Calculate chunk boundaries for sender
                    send_start = min(chunk_idx * max_elements, send_array.size)
                    send_end = min((chunk_idx + 1) * max_elements, send_array.size)

                    # Calculate chunk boundaries for receiver
                    recv_start = min(chunk_idx * max_elements, actual_recv_buffer.size)
                    recv_end = min(
                        (chunk_idx + 1) * max_elements, actual_recv_buffer.size
                    )

                    send_chunk_size = send_end - send_start
                    recv_chunk_size = recv_end - recv_start

                    # Only send/recv if there's actual data
                    if recv_chunk_size > 0:
                        recv_chunk = recv_flat[recv_start:recv_end]
                        req_recv = comm.Irecv(recv_chunk, source=recv_from)
                    else:
                        req_recv = None

                    if send_chunk_size > 0:
                        send_chunk = send_flat[send_start:send_end]
                        req_send = comm.Isend(send_chunk, dest=send_to)
                    else:
                        req_send = None

                    if req_recv is not None:
                        req_recv.Wait()

                    if req_send is not None:
                        req_send.Wait()
            else:
                req_recv = comm.Irecv(actual_recv_buffer, source=recv_from)
                req_send = comm.Isend(send_array, dest=send_to)

                req_recv.Wait()
                req_send.Wait()

            # If we used a temporary buffer, copy it to the actual output location
            if use_temp_buffer:
                output_array[tuple(recv_slices)] = temp_recv

    return output_array


def alltoall(arrays: List[np.ndarray], comm: MPI.Comm) -> List[np.ndarray]:
    """Distributes a list of contiguous numpy arrays from each rank to every other rank.

    It also handles the case where the array sizes are larger than the max allowed by MPI
    (INT_MAX elements, i.e. 2*31), since all MPI calls use the C int data type for representing
    sizes.
    It fixes this with reslice in mind, so the input arrays in the list must:

    - be 3-dimensional
    - One of these dimensions must be the same lengths for all arrays across sent/received arrays
      (reslice maps from current slice dim to next slice dim and leaves the third dimension
      untouched)

    It picks this consistently-sized dimension and creates a new contiguous MPI data type
    of that length. Then the sizes are divided by this length, which should make it fit in all
    practical cases. If not, MPI will raise an exception.

    Parameters
    ----------
    arrays : List[np.ndarray]
        List of 3D numpy arrays to be distributed. Length must be the full size of the given
        communicator.

    Returns
    -------
    List[np.ndarray]
        List of the numpy arrays received. Length is the full size of the given communicator.
    """

    if len(arrays) != comm.size:
        err_str = "list of arrays for MPI alltoall call must match communicator size"
        raise ValueError(err_str)

    assert all(type(a) == np.ndarray for a in arrays), "All arrays must be numpy arrays"
    assert all(
        a.dtype == arrays[0].dtype for a in arrays
    ), "All arrays must be of the same type"
    assert arrays[0].dtype in [
        np.float32,
        np.uint16,
    ], "Only 16bit unsigned ints or single precision floats are implemented"
    assert all(a.ndim == 3 for a in arrays), "Only 3D arrays are supported"

    # no MPI or only one process
    if comm.size == 1:
        return arrays

    sizes_send = [a.size for a in arrays]
    shapes_send = [a.shape for a in arrays]

    # create a single contiguous array with all the arrays flattened and stacked up,
    # so that we can use MPI's Alltoallv (with buffer pointer + offsets)
    # Note: the returned array from concatenate appears to always be C-contiguous
    fullinput = np.concatenate([a.reshape(a.size) for a in arrays])
    assert fullinput.flags.c_contiguous, "C-contigous array is required"
    dtype = MPI.FLOAT if arrays[0].dtype == np.float32 else MPI.UINT16_T

    # let everyone know the shapes / sizes they are going to receive + create an output buffer
    shapes_rec = comm.alltoall(shapes_send)
    sizes_rec = [np.prod(sh) for sh in shapes_rec]
    fulloutput = np.empty((np.sum(sizes_rec),), dtype=arrays[0].dtype)

    # NOTE: The custom MPI data type is being used below even when the number of elements
    # doesn't exceed the limit which can be sent in a single MPI operation. See issue #274.

    # find the dim which is equal in all arrays to send/receive
    dim0s = [s[0] for s in shapes_send] + [s[0] for s in shapes_rec]
    dim1s = [s[1] for s in shapes_send] + [s[1] for s in shapes_rec]
    dim2s = [s[2] for s in shapes_send] + [s[2] for s in shapes_rec]
    dim0equal = all(s == dim0s[0] for s in dim0s)
    dim1equal = all(s == dim1s[1] for s in dim1s)
    dim2equal = all(s == dim2s[2] for s in dim2s)
    assert (
        dim0equal or dim1equal or dim2equal
    ), "At least one dimension of the input arrays must be of same size"

    # create a new contiguous MPI datatype by repeating the input type by this common length
    factor = (
        arrays[0].shape[0]
        if dim0equal
        else arrays[0].shape[1] if dim1equal else arrays[0].shape[2]
    )
    dtype1 = dtype.Create_contiguous(factor).Commit()
    # sanity check - this should always pass
    assert all(s % factor == 0 for s in sizes_send), "Size does not divide evenly"
    assert all(s % factor == 0 for s in sizes_rec), "Size does not divide evenly"
    sizes_send1 = [s // factor for s in sizes_send]
    sizes_rec1 = [s // factor for s in sizes_rec]

    # now send the same data, but with the adjusted size+datatype (output is identical)
    comm.Alltoallv((fullinput, sizes_send1, dtype1), (fulloutput, sizes_rec1, dtype1))

    # build list of output arrays
    cumsizes = np.cumsum(sizes_rec)
    cumsizes = [0, *cumsizes[:-1]]
    ret = list()
    for i, s in enumerate(cumsizes):
        ret.append(fulloutput[s : s + sizes_rec[i]].reshape(shapes_rec[i]))

    return ret
