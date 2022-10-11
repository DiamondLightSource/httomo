import cupy
from tomopy import astra, recon


def reconstruct(
    data: cupy.ndarray,
    angles_radians: cupy.ndarray,
    rot_center: cupy.ndarray,
    use_GPU: int,
):
    """Perform a reconstruction using tomopy's astra recon function.

    Args:
        data: The sinograms to reconstruct.
        angles_radians: A numpy array of angles in radians.
        rot_center: The rotational center.
        use_GPU: The ID of the GPU to use.

    Returns:
        ndarray: A numpy array containing the reconstructed volume.
    """
    data = cupy.asnumpy(data)
    angles_radians = cupy.asnumpy(angles_radians)
    rot_center = cupy.asnumpy(rot_center)

    return recon(
        data,
        angles_radians,
        center=rot_center,
        algorithm=astra,
        options={
            "method": "FBP_CUDA",
            "proj_type": "cuda",
            "gpu_list": [use_GPU],
        },
        ncore=1,
    )
