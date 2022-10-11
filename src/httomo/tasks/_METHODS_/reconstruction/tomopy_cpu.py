from numpy import ndarray, swapaxes
from tomopy import recon


def reconstruct(data: ndarray, angles_radians: ndarray, rot_center: float) -> ndarray:
    """Perform a reconstruction using tomopy's recon function.

    Args:
        data: The sinograms to reconstruct.
        angles_radians: A numpy array of angles in radians.
        rot_center: The rotational center.

    Returns:
        ndarray: A numpy array containing the reconstructed volume.
    """
    return recon(
        swapaxes(data, 0, 1),
        angles_radians,
        center=rot_center,
        algorithm="gridrec",
        sinogram_order=True,
        ncore=1,
    )
