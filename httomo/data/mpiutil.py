from typing import List

import numpy as np
from mpi4py import MPI


__all__ = ["alltoall"]


# add this here so that we can mock it in the tests
_mpi_max_elements = 2**31


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
