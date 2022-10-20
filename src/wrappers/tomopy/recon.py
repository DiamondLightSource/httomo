from numpy import ndarray, swapaxes
from tomopy import recon
from mpi4py.MPI import Comm


def algorithm(data: ndarray, method_name: str, angles_radians: ndarray,
              rot_center: float) -> ndarray:
    """Wrapper for tomopy.recon.algorithm module.

    Args:
        data: The sinograms to reconstruct.
        method_name: The name of the method to use
        angles_radians: A numpy array of angles in radians.
        rot_center: The rotational center.

    Returns:
        ndarray: A numpy array containing the reconstructed volume.
    """
    module = getattr(recon, 'algorithm')
    return getattr(module, 'recon')(
        swapaxes(data, 0, 1),
        angles_radians,
        center=rot_center,
        algorithm=method_name,
        sinogram_order=True,
        ncore=1,
    )


def rotation(data: ndarray, method_name:str, comm: Comm) -> float:
    """Wrapper for the tomopy.recon.rotation module.

    Args:
        data: A numpy array of projections.
        method_name: The name of the method to use
        comm: The MPI communicator to be used.

    Returns:
        float:  The center of rotation.
    """
    module = getattr(recon, 'rotation')
    method_func = getattr(module, method_name)
    rot_center = 0
    mid_rank = int(round(comm.size / 2) + 0.1)
    if comm.rank == mid_rank:
        mid_slice = data.shape[1] // 2
        rot_center = method_func(data[:, mid_slice, :], step=0.5, ncore=1)
    rot_center = comm.bcast(rot_center, root=mid_rank)

    return rot_center
