from mpi4py import MPI

from httomo.utils import gpu_enabled, xp


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
