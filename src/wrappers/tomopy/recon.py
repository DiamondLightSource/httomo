from typing import Dict

from numpy import ndarray, swapaxes
from mpi4py.MPI import Comm
# TODO: Doing `from tomopy import recon` imports the function
# `tomopy.recon.algorithm.recon()` rather than the module `tomopy.recon`, so the
# two lines below are a temporary workaround to this
from importlib import import_module
recon = import_module('tomopy.recon')


def algorithm(params: Dict, method_name: str, data: ndarray,
              angles_radians: ndarray) -> ndarray:
    """Wrapper for tomopy.recon.algorithm module.

    Args:
        params: A dict containing all params of the wrapped tomopy function that
                are not related to the data loaded by a loader function
        method_name: The name of the method to use in tomopy.recon.algorithm
        data: The sinograms to reconstruct.
        angles_radians: A numpy array of angles in radians.

    Returns:
        ndarray: A numpy array containing the reconstructed volume.
    """
    module = getattr(recon, 'algorithm')
    return getattr(module,method_name)(
        swapaxes(data, 0, 1),
        angles_radians,
        **params
    )


def rotation(params: Dict, method_name:str, comm: Comm, data: ndarray) -> float:
    """Wrapper for the tomopy.recon.rotation module.

    Args:
        params: A dict containing all params of the wrapped tomopy function that
                are not related to the data loaded by a loader function
        method_name: The name of the method to use in tomopy.recon.rotation
        comm: The MPI communicator to be used.
        data: A numpy array of projections.

    Returns:
        float:  The center of rotation.
    """
    module = getattr(recon, 'rotation')
    method_func = getattr(module, method_name)
    rot_center = 0
    mid_rank = int(round(comm.size / 2) + 0.1)
    if comm.rank == mid_rank:
        mid_slice = data.shape[1] // 2
        rot_center = method_func(data[:, mid_slice, :], **params)
    rot_center = comm.bcast(rot_center, root=mid_rank)

    return rot_center
