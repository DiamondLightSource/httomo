from typing import Optional

import cupy
from cupyx.scipy.ndimage import gaussian_filter, shift


def find_center_of_rotation(data: cupy.ndarray) -> float:
    """Find the center of rotation using Youself's kernel.

    Args:
        data: A cupy array of projections.

    Returns:
        float: The center of rotation.
    """
    return _find_center_vo_gpu(data)


def _find_center_vo_gpu(
    sino: cupy.ndarray,
    ind: Optional[int] = None,
    smin: int = -50,
    smax: int = 50,
    srad: int = 6,
    step: float = 0.25,
    ratio: float = 0.5,
    drop: int = 20,
):
    if sino.ndim == 2:
        sino = cupy.expand_dims(sino, 1)
        ind = 0
    (depth, height, width) = sino.shape

    if ind is None:
        ind = height // 2
        if height > 10:
            _sino = cupy.mean(sino[:, ind - 5 : ind + 5, :], axis=1)
        else:
            _sino = sino[:, ind, :]
    else:
        _sino = sino[:, ind, :]

    _sino_cs = gaussian_filter(_sino, (3, 1), mode="reflect")
    _sino_fs = gaussian_filter(_sino, (2, 2), mode="reflect")

    if _sino.shape[0] * _sino.shape[1] > 4e6:
        # data is large, so downsample it before performing search for
        # centre of rotation
        _sino_coarse = _downsample(cupy.expand_dims(_sino_cs, 1), 2, 2)[:, 0, :]
        init_cen = _search_coarse(_sino_coarse, smin / 4.0, smax / 4.0, ratio, drop)
        fine_cen = _search_fine(_sino_fs, srad, step, init_cen * 4.0, ratio, drop)
    else:
        init_cen = _search_coarse(_sino_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_sino_fs, srad, step, init_cen, ratio, drop)

    return fine_cen


def _search_coarse(sino, smin, smax, ratio, drop):
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    smin_clip_val = cupy.clip(cupy.asarray([smin + cen_fliplr]), 0, ncol - 1)
    smin = cupy.float(smin_clip_val - cen_fliplr)
    smax_clip_val = cupy.clip(cupy.asarray([smax + cen_fliplr]), 0, ncol - 1)
    smax = cupy.float(smax_clip_val - cen_fliplr)
    start_cor = ncol // 2 + smin
    stop_cor = ncol // 2 + smax
    flip_sino = cupy.fliplr(sino)
    comp_sino = cupy.flipud(sino)
    list_cor = cupy.arange(start_cor, stop_cor + 0.5, 0.5)
    list_metric = cupy.zeros(len(list_cor), dtype=cupy.float32)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = cupy.zeros(list_shift.shape, dtype="float32")
    # TODO: for now, use a for loop to calculate the metric for each shift, but
    # this should be optimised in the future, as it's a computationally
    # expensive part of the algorithm
    for i in range(list_shift.shape[0]):
        list_metric[i] = _calculate_metric(
            list_shift[i], sino, flip_sino, comp_sino, mask
        )

    minpos = cupy.argmin(list_metric)
    if minpos == 0:
        print("WARNING!!!Global minimum is out of searching range")
        print(f"Please extend smin: {smin}")
    if minpos == len(list_metric) - 1:
        print("WARNING!!!Global minimum is out of searching range")
        print(f"Please extend smax: {smax}")
    cor = list_cor[minpos]
    return cor


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    srad_clip_val = cupy.clip(cupy.asarray([cupy.abs(srad)]), 1.0, ncol / 4.0)
    srad = cupy.float(srad_clip_val)
    step_clip_val = cupy.clip(cupy.asarray([cupy.abs(step)]), 0.1, srad)
    step = cupy.float(step_clip_val)
    init_cen_clip_val = cupy.clip(cupy.asarray([init_cen]), srad, ncol - srad - 1)
    init_cen = cupy.float(init_cen_clip_val)
    list_cor = init_cen + cupy.arange(-srad, srad + step, step)
    flip_sino = cupy.fliplr(sino)
    comp_sino = cupy.flipud(sino)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = cupy.zeros(list_shift.shape, dtype="float32")
    # TODO: for now, use a for loop to calculate the metric for each shift, but
    # this should be optimised in the future, as it's a computationally
    # expensive part of the algorithm
    for i in range(list_shift.shape[0]):
        list_metric[i] = _calculate_metric(
            list_shift[i], sino, flip_sino, comp_sino, mask
        )
    cor = list_cor[cupy.argmin(list_metric)]
    return cor


def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * cupy.pi)
    cen_row = int(cupy.ceil(nrow / 2.0) - 1)
    cen_col = int(cupy.ceil(ncol / 2.0) - 1)
    drop = cupy.min(cupy.asarray([drop, int(cupy.ceil(0.05 * nrow))]))

    generate_mask_kernel = cupy.RawKernel(
        r"""
    extern "C" __global__
    void generate_mask(const int ncol, const int nrow, const int cen_col,
                       const int cen_row, const float du, const float dv,
                       const float radius, unsigned short* mask) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int tid = (j * ncol) + i;
        int pos, temp;
        int pos1, pos2;

        if (tid < ncol * nrow) {
            pos = __float2int_ru(((j - cen_row) * dv / radius) / du);
            pos1 = -pos + cen_col;
            pos2 = pos + cen_col;

            if (pos1 > pos2) {
                temp = pos1;
                pos1 = pos2;
                pos2 = temp;
                if (pos1 > ncol - 1) {
                    pos1 = ncol -1;
                }
                if (pos2 < 0) {
                    pos2 = 0;
                }
            }
            else{
                if (pos1 < 0) {
                    pos1 = 0;
                }

                if (pos2 > ncol - 1) {
                    pos2 = ncol -1;
                }
            }

            if (pos1 <= i && i <= pos2) {
                mask[tid] = 1;
            }
        }
    }
    """,
        "generate_mask",
    )

    block_x = 8
    block_y = 8
    block_dims = (block_x, block_y)
    grid_x = int(cupy.ceil(ncol / block_x))
    grid_y = int(cupy.ceil(nrow / block_y))
    grid_dims = (grid_x, grid_y)
    mask = cupy.zeros((nrow, ncol), dtype="uint16")
    params = (
        ncol,
        nrow,
        cen_col,
        cen_row,
        cupy.float32(du),
        cupy.float32(dv),
        cupy.float32(radius),
        mask,
    )
    generate_mask_kernel(grid_dims, block_dims, params)
    mask[cen_row - drop : cen_row + drop + 1, :] = 0
    mask[:, cen_col - 1 : cen_col + 2] = 0
    return mask


def _calculate_metric(shift_col, sino1, sino2, sino3, mask):
    # TODO: currently this function is passed one element at a time from the
    # caller's shift_col array inside a for loop; this is inefficient, and this
    # function should be considered to be rewritten as a kernel function for
    # better performance
    if cupy.abs(shift_col - cupy.floor(shift_col)) == 0.0:
        shift_col = int(shift_col)
        sino_shift = cupy.roll(sino2, shift_col, axis=1)
        if shift_col >= 0:
            sino_shift[:, :shift_col] = sino3[:, :shift_col]
        else:
            sino_shift[:, shift_col:] = sino3[:, shift_col:]
        mat = cupy.vstack((sino1, sino_shift))
    else:
        sino_shift = shift(sino2, (0, shift_col), order=3, prefilter=True)
        if shift_col >= 0:
            shift_int = int(cupy.ceil(shift_col))
            sino_shift[:, :shift_int] = sino3[:, :shift_int]
        else:
            shift_int = int(cupy.floor(shift_col))
            sino_shift[:, shift_int:] = sino3[:, shift_int:]
        mat = cupy.vstack((sino1, sino_shift))
    metric = cupy.mean(cupy.abs(cupy.fft.fftshift(cupy.fft.fft2(mat)) * mask))

    return cupy.asarray([metric], dtype="float32")


DOWNSAMPLE_SINO_KERNEL = cupy.RawKernel(
    r"""
    extern "C" __global__
    void downsample_sino(float* sino, int dx, int dy, int dz, int level,
                         float* out) {
        // use shared memory to store the values used to "merge" columns of the
        // sinogram in the downsampling process
        extern __shared__ float downsampled_vals[];

        unsigned int binsize, i, j, k, orig_ind, out_ind, output_bin_no;
        i = blockDim.x * blockIdx.x + threadIdx.x;
        j = 0;
        k = blockDim.y * blockIdx.y + threadIdx.y;
        orig_ind = (k * dz) + i;

        binsize = __float2uint_rd(
            powf(__uint2float_rd(2), level));
        unsigned int dz_downsampled = __float2uint_rd(
            fdividef(__uint2float_rd(dz), __uint2float_rd(binsize)));
        unsigned int i_downsampled = __float2uint_rd(
            fdividef(__uint2float_rd(i), __uint2float_rd(binsize)));

        if (orig_ind < dx * dz) {
            output_bin_no =  __float2uint_rd(
                fdividef(__uint2float_rd(i), __uint2float_rd(binsize))
            );

            out_ind = (k * dz_downsampled) + i_downsampled;
            downsampled_vals[threadIdx.y * 8 + threadIdx.x] = sino[orig_ind] / __uint2float_rd(binsize);
            // synchronise threads within thread-block so that it's guaranteed
            // that all the required values have been copied into shared memeory
            // to then sum and save in the downsampled output
            __syncthreads();

            // arbitrarily use the "beginning thread" in each "lot" of pixels
            // for downsampling to then save the desired value in the
            // downsampled output array
            if (i % 4 == 0) {
                out[out_ind] = downsampled_vals[threadIdx.y * 8 + threadIdx.x] +
                    downsampled_vals[threadIdx.y * 8 + threadIdx.x + 1] +
                    downsampled_vals[threadIdx.y * 8 + threadIdx.x + 2] +
                    downsampled_vals[threadIdx.y * 8 + threadIdx.x + 3];
            }
        }
    }
    """,
    "downsample_sino",
)


def _downsample(sino, level, axis):
    sino = cupy.asarray(sino, dtype="float32")
    dx, dy, dz = sino.shape
    # Determine the new size, dim, of the downsampled dimension
    dim = int(sino.shape[axis] / cupy.power(2, level))
    shape = [dx, dy, dz]
    shape[axis] = dim
    downsampled_data = cupy.zeros(shape, dtype="float32")

    block_x = 8
    block_y = 8
    block_dims = (block_x, block_y)
    grid_x = int(cupy.ceil(sino.shape[2] / block_x))
    grid_y = int(cupy.ceil(sino.shape[0] / block_y))
    grid_dims = (grid_x, grid_y)
    # 8x8 thread-block, which means 16 "lots" of columns to downsample per
    # thread-block; 4 bytes per float, so allocate 16*6 = 64 bytes of shared
    # memeory per thread-block
    shared_mem_bytes = 64
    params = (sino, dx, dy, dz, level, downsampled_data)
    DOWNSAMPLE_SINO_KERNEL(grid_dims, block_dims, params, shared_mem=shared_mem_bytes)
    return downsampled_data
