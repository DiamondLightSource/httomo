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
