from httomo.utils import Colour, log_once


gpu_enabled = False

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
        gpu_enabled = True  # CuPy is installed and GPU is available
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        print("CuPy is installed but GPU device inaccessible")

except ImportError:
    import numpy as xp

    log_once(
        "CuPy is not installed",
        # The `comm` parameter for `log_once()` isn't used in the function body,
        # so any value is fine to pass
        comm="comm",
        colour=Colour.LYELLOW,
        level=1,
    )


def _get_available_gpu_memory(safety_margin_percent: float = 10.0) -> int:
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        # first, let's make some space
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        cache = cp.fft.config.get_plan_cache()
        cache.clear()
        available_memory = dev.mem_info[0] + pool.free_bytes()
        return int(available_memory * (1 - safety_margin_percent / 100.0))
    except:
        return int(100e9)  # arbitrarily high number - only used if GPU isn't available
