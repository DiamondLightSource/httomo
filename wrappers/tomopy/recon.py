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

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of HTTomo.
    method_name : str
        The name of the method to use in tomopy.recon.algorithm
    data : ndarray
        The sinograms to reconstruct.
    angles_radians : ndarray
        A numpy array of angles in radians.

    Returns
    -------
    ndarray
        A numpy array containing the reconstructed volume.
    """
    module = getattr(recon, 'algorithm')
    return getattr(module,method_name)(
        data,
        angles_radians,
        **params
    )


def rotation(params: Dict, method_name:str, comm: Comm, data: ndarray) -> float:
    """Wrapper for the tomopy.recon.rotation module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of HTTomo.
    method_name : str
        The name of the method to use in tomopy.recon.rotation.
    comm : Comm
        The MPI communicator to be used.
    data : ndarray
        A numpy array of projections.

    Returns
    -------
    float
        The center of rotation.
    """
   
    module = getattr(recon, 'rotation')
    method_func = getattr(module, method_name)
    rot_center = 0
    mid_rank = int(round(comm.size / 2) + 0.1)
    if comm.rank == mid_rank:
        if params['ind'] == 'mid':
            params['ind'] = data.shape[1] // 2 # get the middle slice
        rot_center = method_func(data, **params)
    rot_center = comm.bcast(rot_center, root=mid_rank)

    return rot_center
