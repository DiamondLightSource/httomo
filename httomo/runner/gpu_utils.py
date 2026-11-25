from mpi4py import MPI

from httomo.utils import gpu_enabled, xp
import numpy as np


def gpumem_cleanup():
    """cleans up GPU memory and also the FFT plan cache"""
    if gpu_enabled:
        xp.get_default_memory_pool().free_all_blocks()
        cache = xp.fft.config.get_plan_cache()
        cache.clear()


def get_available_gpu_memory(safety_margin_percent: float = 10.0) -> int:
    try:
        import cupy as cp

        dev = cp.cuda.Device(get_gpu_id())
        with dev:
            gpumem_cleanup()
            pool = cp.get_default_memory_pool()
            available_memory = dev.mem_info[0] + pool.free_bytes()
            return int(available_memory * (1 - safety_margin_percent / 100.0))
    except:
        return int(100e9)  # arbitrarily high number - only used if GPU isn't available


def get_gpu_id() -> int:
    """
    Get the ID of the specific GPU on the machine that the process should use
    """
    num_gpus = xp.cuda.runtime.getDeviceCount()
    global_comm = MPI.COMM_WORLD
    local_comm = global_comm.Split_type(MPI.COMM_TYPE_SHARED)
    return local_comm.rank % num_gpus


def raw_memcpy_h2d(h_data: np.ndarray) -> xp.ndarray:
    """
    Blocking copy from the host to the CUDA device, on the current stream.
    This function uses the CUDA runtime API directly, without CuPy's management
    or pinned memory. The input array must be C contiguous.
    """
    assert gpu_enabled
    assert h_data.flags.c_contiguous
    d_data = xp.empty_like(h_data)
    xp.cuda.runtime.memcpyAsync(
        d_data.data.ptr,
        h_data.ctypes.data,
        d_data.nbytes,
        xp.cuda.runtime.memcpyHostToDevice,
        xp.cuda.get_current_stream().ptr,
    )
    xp.cuda.get_current_stream().synchronize()
    return d_data


def raw_memcpy_d2h(d_data: xp.ndarray) -> np.ndarray:
    """
    Blocking copy from the CUDA device to the host, on the current stream.
    This function uses the CUDA runtime API directly, without CuPy's management
    or pinned memory. The input array must be C contiguous.
    """
    assert gpu_enabled
    assert d_data.flags.c_contiguous
    h_data = np.empty_like(d_data)
    xp.cuda.runtime.memcpyAsync(
        h_data.data.ptr,
        d_data.ctypes.data,
        h_data.nbytes,
        xp.cuda.runtime.memcpyDeviceToHost,
        xp.cuda.get_current_stream().ptr,
    )
    xp.cuda.get_current_stream().synchronize()
    return h_data
