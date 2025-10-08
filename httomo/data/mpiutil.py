from typing import List

import logging

import numpy as np
from mpi4py import MPI

from httomo.utils import log_once


__all__ = ["alltoall", "alltoall_ring"]


# add this here so that we can mock it in the tests
_mpi_max_elements = 2**31


def alltoall_ring(arrays: List[np.ndarray], comm: MPI.Comm, concat_axis: int = 0) -> np.ndarray:
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
        f"[Rank {rank}] alltoall_ring: Starting with {len(arrays)} arrays, comm.size={nprocs}, concat_axis={concat_axis}",
        level=logging.DEBUG,
    )

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
        log_once(
            f"[Rank {rank}] alltoall_ring: Single process, returning first array as-is",
            level=logging.DEBUG,
        )
        return arrays[0]

    shapes_send = [a.shape for a in arrays]

    log_once(
        f"[Rank {rank}] alltoall_ring: Array shapes to send: {shapes_send}",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] alltoall_ring: Array sizes to send: {[a.size for a in arrays]}",
        level=logging.DEBUG,
    )

    # Exchange shape information first
    log_once(
        f"[Rank {rank}] alltoall_ring: Exchanging shape information via alltoall",
        level=logging.DEBUG,
    )
    shapes_rec = comm.alltoall(shapes_send)
    log_once(
        f"[Rank {rank}] alltoall_ring: Received shapes: {shapes_rec}",
        level=logging.DEBUG,
    )

    # Calculate output shape for concatenated array
    output_shape = list(shapes_rec[0])
    output_shape[concat_axis] = sum(s[concat_axis] for s in shapes_rec)

    log_once(
        f"[Rank {rank}] alltoall_ring: Pre-allocating concatenated output array with shape {tuple(output_shape)}",
        level=logging.DEBUG,
    )

    # Pre-allocate the final concatenated output array
    output_array = np.empty(output_shape, dtype=arrays[0].dtype)

    # Calculate offsets for each rank's data in the output array
    offsets = [0]
    for i in range(nprocs - 1):
        offsets.append(offsets[-1] + shapes_rec[i][concat_axis])

    log_once(
        f"[Rank {rank}] alltoall_ring: Offsets along axis {concat_axis}: {offsets}",
        level=logging.DEBUG,
    )

    # Use ring pattern: each process sends to (rank + offset) and receives from (rank - offset)
    for offset in range(nprocs):
        send_to = (rank + offset) % nprocs
        recv_from = (rank - offset) % nprocs

        log_once(
            f"[Rank {rank}] alltoall_ring: Ring iteration {offset}/{nprocs-1}, send_to={send_to}, recv_from={recv_from}",
            level=logging.DEBUG,
        )

        # Create slice for where to write received data
        recv_slices = [slice(None)] * 3
        recv_offset = offsets[recv_from]
        recv_size = shapes_rec[recv_from][concat_axis]
        recv_slices[concat_axis] = slice(recv_offset, recv_offset + recv_size)

        if send_to == rank:
            # Self-copy (no communication)
            log_once(
                f"[Rank {rank}] alltoall_ring: Self-copy for rank {rank}, shape={arrays[rank].shape}, "
                f"writing to offset {recv_offset}:{recv_offset + recv_size}",
                level=logging.DEBUG,
            )
            output_array[tuple(recv_slices)] = arrays[rank]
        else:
            # Note: We send arrays[send_to] TO send_to, and receive FROM recv_from
            send_array = arrays[send_to]
            if not send_array.flags.c_contiguous:
                log_once(
                    f"[Rank {rank}] alltoall_ring: Converting non-contiguous array to contiguous for send_to {send_to}",
                    level=logging.DEBUG,
                )
                send_array = np.ascontiguousarray(send_array)

            # Get view into output array for receiving
            recv_view = output_array[tuple(recv_slices)]

            log_once(
                f"[Rank {rank}] alltoall_ring: Communicating send_to={send_to}, recv_from={recv_from}, "
                f"send_size={send_array.size}, recv_size={recv_view.size}, "
                f"recv_offset={recv_offset}",
                level=logging.DEBUG,
            )

            # Chunk if array is too large
            max_elements = _mpi_max_elements
            if send_array.size > max_elements:
                # Send in chunks
                log_once(
                    f"[Rank {rank}] alltoall_ring: Array too large ({send_array.size} > {max_elements}), "
                    f"using chunked transfer send_to={send_to}, recv_from={recv_from}",
                    level=logging.DEBUG,
                )
                send_flat = send_array.ravel()
                recv_flat = recv_view.ravel()
                num_chunks = (send_array.size + max_elements - 1) // max_elements

                log_once(
                    f"[Rank {rank}] alltoall_ring: Splitting into {num_chunks} chunks send_to={send_to}, recv_from={recv_from}",
                    level=logging.DEBUG,
                )

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * max_elements
                    end = min(start + max_elements, send_array.size)

                    log_once(
                        f"[Rank {rank}] alltoall_ring: Chunk {chunk_idx}/{num_chunks-1} send_to={send_to}, recv_from={recv_from}, "
                        f"elements [{start}:{end}]",
                        level=logging.DEBUG,
                    )

                    send_chunk = send_flat[start:end]
                    recv_chunk = recv_flat[start:end]

                    log_once(
                        f"[Rank {rank}] alltoall_ring: Posting Irecv for chunk {chunk_idx} from {recv_from}",
                        level=logging.DEBUG,
                    )
                    req_recv = comm.Irecv(recv_chunk, source=recv_from)

                    log_once(
                        f"[Rank {rank}] alltoall_ring: Posting Isend for chunk {chunk_idx} to {send_to}",
                        level=logging.DEBUG,
                    )
                    req_send = comm.Isend(send_chunk, dest=send_to)

                    log_once(
                        f"[Rank {rank}] alltoall_ring: Waiting for Irecv completion for chunk {chunk_idx} from {recv_from}",
                        level=logging.DEBUG,
                    )
                    req_recv.Wait()

                    log_once(
                        f"[Rank {rank}] alltoall_ring: Waiting for Isend completion for chunk {chunk_idx} to {send_to}",
                        level=logging.DEBUG,
                    )
                    req_send.Wait()

                    log_once(
                        f"[Rank {rank}] alltoall_ring: Chunk {chunk_idx} transfer complete (send_to={send_to}, recv_from={recv_from})",
                        level=logging.DEBUG,
                    )
            else:
                log_once(
                    f"[Rank {rank}] alltoall_ring: Posting Irecv from {recv_from}",
                    level=logging.DEBUG,
                )
                req_recv = comm.Irecv(recv_view, source=recv_from)

                log_once(
                    f"[Rank {rank}] alltoall_ring: Posting Isend to {send_to}",
                    level=logging.DEBUG,
                )
                req_send = comm.Isend(send_array, dest=send_to)

                log_once(
                    f"[Rank {rank}] alltoall_ring: Waiting for Irecv completion from {recv_from}",
                    level=logging.DEBUG,
                )
                req_recv.Wait()

                log_once(
                    f"[Rank {rank}] alltoall_ring: Waiting for Isend completion to {send_to}",
                    level=logging.DEBUG,
                )
                req_send.Wait()

                log_once(
                    f"[Rank {rank}] alltoall_ring: Communication complete (send_to={send_to}, recv_from={recv_from})",
                    level=logging.DEBUG,
                )

            log_once(
                f"[Rank {rank}] alltoall_ring: Received data from {recv_from} written to output at offset {recv_offset}",
                level=logging.DEBUG,
            )

    log_once(
        f"[Rank {rank}] alltoall_ring: Ring communication completed successfully, returning concatenated array",
        level=logging.DEBUG,
    )

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
    rank = comm.rank
    nprocs = comm.size

    log_once(
        f"[Rank {rank}] alltoall: Starting with {len(arrays)} arrays, comm.size={nprocs}",
        level=logging.DEBUG,
    )

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
        log_once(
            f"[Rank {rank}] alltoall: Single process, returning arrays as-is",
            level=logging.DEBUG,
        )
        return arrays

    sizes_send = [a.size for a in arrays]
    shapes_send = [a.shape for a in arrays]

    log_once(
        f"[Rank {rank}] alltoall: Array shapes to send: {shapes_send}",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] alltoall: Array sizes to send: {sizes_send}",
        level=logging.DEBUG,
    )

    # create a single contiguous array with all the arrays flattened and stacked up,
    # so that we can use MPI's Alltoallv (with buffer pointer + offsets)
    # Note: the returned array from concatenate appears to always be C-contiguous
    log_once(
        f"[Rank {rank}] alltoall: Concatenating input arrays",
        level=logging.DEBUG,
    )
    fullinput = np.concatenate([a.reshape(a.size) for a in arrays])
    assert fullinput.flags.c_contiguous, "C-contigous array is required"
    dtype = MPI.FLOAT if arrays[0].dtype == np.float32 else MPI.UINT16_T

    # let everyone know the shapes / sizes they are going to receive + create an output buffer
    log_once(
        f"[Rank {rank}] alltoall: Exchanging shape information via alltoall",
        level=logging.DEBUG,
    )
    shapes_rec = comm.alltoall(shapes_send)
    sizes_rec = [np.prod(sh) for sh in shapes_rec]

    log_once(
        f"[Rank {rank}] alltoall: Received shapes: {shapes_rec}",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] alltoall: Received sizes: {sizes_rec}",
        level=logging.DEBUG,
    )

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

    log_once(
        f"[Rank {rank}] alltoall: Creating custom MPI datatype with factor={factor} "
        f"(dim0equal={dim0equal}, dim1equal={dim1equal}, dim2equal={dim2equal})",
        level=logging.DEBUG,
    )

    dtype1 = dtype.Create_contiguous(factor).Commit()
    # sanity check - this should always pass
    assert all(s % factor == 0 for s in sizes_send), "Size does not divide evenly"
    assert all(s % factor == 0 for s in sizes_rec), "Size does not divide evenly"
    sizes_send1 = [s // factor for s in sizes_send]
    sizes_rec1 = [s // factor for s in sizes_rec]

    log_once(
        f"[Rank {rank}] alltoall: Adjusted send sizes: {sizes_send1}",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] alltoall: Adjusted receive sizes: {sizes_rec1}",
        level=logging.DEBUG,
    )

    # now send the same data, but with the adjusted size+datatype (output is identical)
    log_once(
        f"[Rank {rank}] alltoall: Calling MPI Alltoallv",
        level=logging.DEBUG,
    )
    comm.Alltoallv((fullinput, sizes_send1, dtype1), (fulloutput, sizes_rec1, dtype1))

    log_once(
        f"[Rank {rank}] alltoall: MPI Alltoallv completed successfully",
        level=logging.DEBUG,
    )

    # build list of output arrays
    cumsizes = np.cumsum(sizes_rec)
    cumsizes = [0, *cumsizes[:-1]]
    ret = list()
    for i, s in enumerate(cumsizes):
        ret.append(fulloutput[s : s + sizes_rec[i]].reshape(shapes_rec[i]))

    log_once(
        f"[Rank {rank}] alltoall: Completed, returning {len(ret)} arrays",
        level=logging.DEBUG,
    )

    return ret