from cupy import ndarray, swapaxes
from tomobar.methodsDIR import RecToolsDIR


def reconstruct(
    data: ndarray,
    angles_radians: ndarray,
    rot_center: float,
    use_GPU: int,
):
    """Perform a reconstruction using tomobar's recon function.

    Args:
        data: The sinograms to reconstruct.
        angles_radians: A numpy array of angles in radians.
        rot_center: The rotational center.
        use_GPU: The ID of the GPU to use.

    Returns:
        ndarray: A numpy array containing the reconstructed volume.
    """
    RectoolsDIR = RecToolsDIR(
        DetectorsDimH=data.shape[2],  # Horizontal detector dimension
        DetectorsDimV=data.shape[1],
        CenterRotOffset=data.shape[2] * 0.5
        - rot_center,  # Center of Rotation scalar or a vector
        AnglesVec=angles_radians,  # A vector of projection angles
        ObjSize=data.shape[2],  # Reconstructed object dimensions (scalar)
        device_projector=use_GPU,
    )
    return RectoolsDIR.FBP3D_cupy(
        swapaxes(data, 0, 1)
    )  # perform FBP as 3D BP with Astra and then filtering
