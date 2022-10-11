from numpy import ndarray
from mpi4py.MPI import Comm
from tomopy import find_center_vo


def find_center_of_rotation(data: ndarray, comm: Comm) -> float:
    """Finds the center of rotation using tomopy's find_center_vo.

    Args:
        data: A numpy array of projections.
        comm: The MPI communicator to be used.

    Returns:
        float:  The center of rotation.
    """
    rot_center = 0
    mid_rank = int(round(comm.size / 2) + 0.1)
    if comm.rank == mid_rank:
        mid_slice = data.shape[1] // 2
        rot_center = find_center_vo(data[:, mid_slice, :], step=0.5, ncore=1)
    rot_center = comm.bcast(rot_center, root=mid_rank)

    return rot_center
