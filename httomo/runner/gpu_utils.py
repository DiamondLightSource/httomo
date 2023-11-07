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
        from httomo.data.mpiutil import local_rank

        dev = cp.cuda.Device(local_rank)
        # first, let's make some space
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        cache = cp.fft.config.get_plan_cache()
        cache.clear()
        available_memory = dev.mem_info[0] + pool.free_bytes()
        return int(available_memory * (1 - safety_margin_percent / 100.0))
    except:
        return int(100e9)  # arbitrarily high number - only used if GPU isn't available

