import dataclasses
import inspect
import pytest
from mpi4py import MPI
import numpy as np
from numpy import uint16, float32

from numpy.testing import assert_allclose, assert_equal

cupy = pytest.importorskip("cupy")
httomolibgpu = pytest.importorskip("httomolibgpu")
import cupy as cp

from httomolibgpu.prep.normalize import normalize

class MaxMemoryHook(cp.cuda.MemoryHook):
    
    def __init__(self, initial=0):
        self.max_mem = initial
        self.current = initial
    
    def malloc_postprocess(self, device_id: int, size: int, mem_size: int, mem_ptr: int, pmem_id: int):
        self.current += mem_size
        self.max_mem = max(self.max_mem, self.current)

    def free_postprocess(self, device_id: int, mem_size: int, mem_ptr: int, pmem_id: int):
        self.current -= mem_size

    def alloc_preprocess(self, **kwargs):
        pass

    def alloc_postprocess(self, device_id: int, mem_size: int, mem_ptr: int):
        pass
    
    def free_preprocess(self, **kwargs):
        pass

    def malloc_preprocess(self, **kwargs):
        pass

@pytest.mark.cupy
def test_normalize_1D_raises(data, flats, darks, ensure_clean_memory):    
    _data_1d = cp.ones(10)

    #: data cannot be a 1D array
    with pytest.raises(ValueError):
        normalize(_data_1d, flats, darks)

    #: flats cannot be a 1D array
    with pytest.raises(ValueError):
        normalize(data, _data_1d, darks)

@pytest.mark.cupy
def test_normalize(data, flats, darks, ensure_clean_memory):
    # --- testing normalize  ---#
    data_normalize = normalize(cp.copy(data), flats, darks, minus_log=True).get()

    assert data_normalize.dtype == np.float32

    assert_allclose(np.mean(data_normalize), 0.2892469, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 52.064465, rtol=1e-06)
    assert_allclose(np.median(data_normalize), 0.01723744, rtol=1e-06)
    assert_allclose(np.std(data_normalize), 0.524382, rtol=1e-06)

@pytest.mark.cupy
def test_normalize_memoryhook(data, flats, darks, ensure_clean_memory):    
    # --- testing normalize  ---#
    hook = MaxMemoryHook()
    with hook:
        data_normalize = normalize(cp.copy(data), flats, darks, minus_log=True).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook
    max_mem_mb = round(max_mem / (1024**2), 2) # now in mbs
    
    # now we estimate how much of the total memory required for this data
    from httomo.methods_database.query import get_method_info
    library_info = get_method_info("httomolibgpu.prep.normalize", "normalize", "memory_gpu")
    for i, dst in enumerate(library_info[0]['datasets']):
        if dst == "flats":
            flats_bytes = library_info[1]['multipliers'][i] * np.prod(cp.shape(flats)) * float32().nbytes
        elif dst == "darks":
            darks_bytes = library_info[1]['multipliers'][i] * np.prod(cp.shape(darks)) * float32().nbytes
        else:            
            data_bytes = library_info[1]['multipliers'][i] * np.prod(cp.shape(data)) * float32().nbytes
    estimated_memory_bytes = flats_bytes + darks_bytes + data_bytes
    estimated_memory_mb = round(estimated_memory_bytes / (1024**2), 2)
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
    
@pytest.mark.cupy
@pytest.mark.parametrize("slices", [128, 256, 512])
def test_normalize_memoryhook_parametrise(slices, ensure_clean_memory):
    # --- testing normalize  ---#
    data_size_dim = 512
    data = cp.random.random_sample((slices, data_size_dim, data_size_dim), dtype=np.float32)
    darks = cp.random.random_sample((20, data_size_dim, data_size_dim), dtype=np.float32)
    flats = cp.random.random_sample((20, data_size_dim, data_size_dim), dtype=np.float32)
    hook = MaxMemoryHook()
    with hook:
        data_normalize = normalize(cp.copy(data), flats, darks, minus_log=True).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook
    max_mem_mb = round(max_mem / (1024**2), 2) # now in mbs
   
    # now we estimate how much of the total memory required for this data
    from httomo.methods_database.query import get_method_info
    library_info = get_method_info("httomolibgpu.prep.normalize", "normalize", "memory_gpu")
    for i, dst in enumerate(library_info[0]['datasets']):
        if dst == "flats":
            flats_bytes = library_info[1]['multipliers'][i] * np.prod(cp.shape(flats)) * float32().nbytes
        elif dst == "darks":
            darks_bytes = library_info[1]['multipliers'][i] * np.prod(cp.shape(darks)) * float32().nbytes
        else:            
            data_bytes = library_info[1]['multipliers'][i] * np.prod(cp.shape(data)) * float32().nbytes
    estimated_memory_bytes = flats_bytes + darks_bytes + data_bytes
    estimated_memory_mb = round(estimated_memory_bytes / (1024**2), 2)
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
    