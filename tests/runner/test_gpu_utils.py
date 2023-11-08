import pytest
from httomo.data.mpiutil import local_rank
from httomo.runner.gpu_utils import get_available_gpu_memory
from httomo.utils import xp, gpu_enabled


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_get_available_memory():
    mem = get_available_gpu_memory(10.0)

    assert mem > 0
    assert mem <= 0.9 * xp.cuda.Device(local_rank).mem_info[0]


@pytest.mark.skipif(gpu_enabled, reason="This is a CPU-only test")
def test_get_available_memory_cpu():
    # greater 10GB
    assert get_available_gpu_memory() > 10e9
