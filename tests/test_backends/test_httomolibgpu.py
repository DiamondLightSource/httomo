import dataclasses
import inspect
import pytest
from mpi4py import MPI
import numpy as np
from numpy import uint16, float32

from numpy.testing import assert_allclose, assert_equal
import os

cupy = pytest.importorskip("cupy")
httomolibgpu = pytest.importorskip("httomolibgpu")
import cupy as cp

from httomo.methods_database.query import get_method_info

from httomolibgpu.prep.normalize import normalize
from httomolibgpu.prep.phase import paganin_filter_tomopy, paganin_filter_savu
from httomolibgpu.prep.alignment import distortion_correction_proj_discorpy
from httomolibgpu.prep.stripe import remove_stripe_based_sorting, remove_stripe_ti
from httomolibgpu.recon.algorithm import FBP, SIRT, CGLS

from httomo.methods_database.packages.external.httomolibgpu.supporting_funcs.prep.phase import *
from httomo.methods_database.packages.external.httomolibgpu.supporting_funcs.prep.stripe import *
from httomo.methods_database.packages.external.httomolibgpu.supporting_funcs.recon.algorithm import *

module_mem_path = "httomo.methods_database.packages.external."
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
def test_normalize_memoryhook(data, flats, darks, ensure_clean_memory):
    hook = MaxMemoryHook()
    with hook:
        data_normalize = normalize(cp.copy(data), flats, darks, minus_log=True).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook
    max_mem_mb = round(max_mem / (1024**2), 2) # now in mbs
    
    # now we estimate how much of the total memory required for this data
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
    # now compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
   
@pytest.mark.cupy
@pytest.mark.parametrize("slices", [128, 256, 512])
def test_normalize_memoryhook_parametrise(slices, ensure_clean_memory):
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
    # now compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20

@pytest.mark.cupy
@pytest.mark.parametrize("dim_x", [340, 135, 96])
@pytest.mark.parametrize("dim_y", [81, 260, 320])
@pytest.mark.parametrize("slices", [64, 128])
def test_paganin_filter_tomopy_memoryhook(slices, dim_x, dim_y, ensure_clean_memory):    
    data = cp.random.random_sample((slices, dim_y, dim_x), dtype=np.float32)
    hook = MaxMemoryHook()
    with hook:
        _ = paganin_filter_tomopy(cp.copy(data)).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook   
    
    # now we estimate how much of the total memory required for this data
    (estimated_memory_bytes, subtract_bytes) = _calc_memory_bytes_paganin_filter_tomopy((dim_y, dim_x), dtype=np.float32())
    estimated_memory_mb = round(slices*estimated_memory_bytes / (1024**2), 2)
    max_mem -= subtract_bytes
    max_mem_mb = round(max_mem / (1024**2), 2)
    
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20


@pytest.mark.cupy
@pytest.mark.parametrize("slices", [64, 128])
@pytest.mark.parametrize("dim_x", [81, 260, 320])
@pytest.mark.parametrize("dim_y", [340, 135, 96])
def test_paganin_filter_savu_memoryhook(slices, dim_x, dim_y, ensure_clean_memory):    
    data = cp.random.random_sample((slices, dim_x, dim_y), dtype=np.float32)
    kwargs = {}
    kwargs["ratio"] = 250.0
    kwargs["energy"] = 53.0
    kwargs["distance"] = 1.0
    kwargs["resolution"] = 1.28
    kwargs["pad_x"] = 20
    kwargs["pad_y"] = 20
    kwargs["pad_method"] = 'edge'
    kwargs["increment"] = 0.0
    hook = MaxMemoryHook()
    with hook:
        data_filtered = paganin_filter_savu(cp.copy(data),
                                            **kwargs).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook   
    
    # now we estimate how much of the total memory required for this data
    (estimated_memory_bytes, subtract_bytes) = _calc_memory_bytes_paganin_filter_savu((dim_x, dim_y), np.float32(), **kwargs)
    estimated_memory_mb = round(slices*estimated_memory_bytes / (1024**2), 2)
    max_mem -= subtract_bytes
    max_mem_mb = round(max_mem / (1024**2), 2)
    
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
    
    
@pytest.mark.cupy
@pytest.mark.parametrize("slices", [128, 190, 256])
def test_distortion_correction_memoryhook(slices, distortion_correction_path, ensure_clean_memory):
    data_size_dim = 320
    data = cp.random.random_sample((slices, data_size_dim, data_size_dim), dtype=np.float32)
    
    distortion_coeffs_path = os.path.join(
        distortion_correction_path, "distortion-coeffs.txt"
    )
    preview = {"starts": [0, 0], "stops": [data.shape[1], data.shape[2]], "steps": [1, 1]}
    
    hook = MaxMemoryHook()
    with hook:
        data_corrected = distortion_correction_proj_discorpy(cp.copy(data), 
                                                             distortion_coeffs_path,
                                                             preview).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook
    max_mem_mb = round(max_mem / (1024**2), 2) # now in mbs
    
    # now we estimate how much of the total memory required for this data
    library_info = get_method_info("httomolibgpu.prep.alignment", "distortion_correction_proj_discorpy", "memory_gpu")
    estimated_memory_bytes = library_info[1]['multipliers'][0] * np.prod(cp.shape(data)) * float32().nbytes
    estimated_memory_mb = round(estimated_memory_bytes / (1024**2), 2)
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
    
@pytest.mark.cupy
@pytest.mark.parametrize("slices", [128, 256, 320])
def test_remove_stripe_based_sorting_memoryhook(slices, distortion_correction_path, ensure_clean_memory):
    data_size_dim = 300
    data = cp.random.random_sample((data_size_dim, slices, data_size_dim), dtype=np.float32)
    
    hook = MaxMemoryHook()
    with hook:
        data_filtered = remove_stripe_based_sorting(cp.copy(data)).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook
    max_mem_mb = round(max_mem / (1024**2), 2) # now in mbs
    
    # now we estimate how much of the total memory required for this data
    library_info = get_method_info("httomolibgpu.prep.stripe", "remove_stripe_based_sorting", "memory_gpu")
    estimated_memory_bytes = library_info[1]['multipliers'][0] * np.prod(cp.shape(data)) * float32().nbytes
    estimated_memory_mb = round(estimated_memory_bytes / (1024**2), 2)
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
    
@pytest.mark.cupy
@pytest.mark.parametrize("slices", [64, 129])
def test_remove_stripe_ti_memoryhook(slices, ensure_clean_memory):    
    dim_x = 156
    dim_y = 216
    data = cp.random.random_sample((slices, dim_x, dim_y), dtype=np.float32)
    hook = MaxMemoryHook()
    with hook:
        data_filtered = remove_stripe_ti(cp.copy(data)).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook   
    
    # now we estimate how much of the total memory required for this data
    (estimated_memory_bytes, subtract_bytes) = _calc_memory_bytes_remove_stripe_ti((dim_x, dim_y), dtype=np.float32())
    estimated_memory_mb = round(slices*estimated_memory_bytes / (1024**2), 2)
    max_mem -= subtract_bytes
    max_mem_mb = round(max_mem / (1024**2), 2)
    
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
    
@pytest.mark.cupy
@pytest.mark.parametrize("slices", [3, 5, 8])
@pytest.mark.parametrize("recon_size_it", [600, 1200, 2560])
def test_recon_FBP_memoryhook(slices, recon_size_it, ensure_clean_memory):
    data = cp.random.random_sample((1801, slices, recon_size_it), dtype=np.float32)
    kwargs = {
        "angles": np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        "center": 500,
        "recon_size": recon_size_it,
        "recon_mask_radius": 0.8,
    }
    hook = MaxMemoryHook()
    with hook:
        _ = FBP(data, **kwargs)

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook   
    max_mem_mb = round(max_mem / (1024**2), 2)
    
    # now we estimate how much of the total memory required for this data
    (estimated_memory_bytes, subtract_bytes) = _calc_memory_bytes_FBP((1801, recon_size_it), dtype=np.float32(), **kwargs)
    estimated_memory_mb = round(slices*estimated_memory_bytes / (1024**2), 2)
    max_mem -= subtract_bytes
    max_mem_mb = round(max_mem / (1024**2), 2)   
    
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 35
    
    
@pytest.mark.cupy
@pytest.mark.parametrize("slices", [3, 5, 8])
def test_recon_SIRT_memoryhook(slices, ensure_clean_memory):
    data = cp.random.random_sample((1801, slices, 2560), dtype=np.float32)
    recon_size = data.shape[2]
    hook = MaxMemoryHook()
    with hook:
        recon_data = SIRT(
                    data,
                    np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
                    1200,
                    recon_size=recon_size,
                    iterations=2,
                    nonnegativity=True,
                    recon_mask_radius = 0.8,
                )

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook   
    max_mem_mb = round(max_mem / (1024**2), 2)
    
    # now we estimate how much of the total memory required for this data
    library_info = get_method_info("httomolibgpu.recon.algorithm", "SIRT", "memory_gpu")
    estimated_memory_bytes = library_info[1]['multipliers'][0] * np.prod(cp.shape(data)) * float32().nbytes
    estimated_memory_mb = round(estimated_memory_bytes / (1024**2), 2)
    
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20


@pytest.mark.cupy
@pytest.mark.parametrize("slices", [3, 5])
def test_recon_CGLS_memoryhook(slices, ensure_clean_memory):
    data = cp.random.random_sample((1801, slices, 2560), dtype=np.float32)
    recon_size = data.shape[2]
    hook = MaxMemoryHook()
    with hook:
        recon_data = CGLS(
                    data,
                    np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
                    1200,
                    recon_size=recon_size,
                    iterations=2,
                    nonnegativity=True,
                    recon_mask_radius = 0.8,
                )

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem # the amount of memory in bytes needed for the method according to memoryhook   
    max_mem_mb = round(max_mem / (1024**2), 2)
    
    # now we estimate how much of the total memory required for this data
    library_info = get_method_info("httomolibgpu.recon.algorithm", "CGLS", "memory_gpu")
    estimated_memory_bytes = library_info[1]['multipliers'][0] * np.prod(cp.shape(data)) * float32().nbytes
    estimated_memory_mb = round(estimated_memory_bytes / (1024**2), 2)
    
    # now we compare both memory estimations 
    difference_mb = abs(estimated_memory_mb - max_mem_mb)
    percents_relative_maxmem = round((difference_mb/max_mem_mb)*100)
    # the estimated_memory_mb should be LARGER or EQUAL to max_mem_mb
    # the resulting percent value should not deviate from max_mem on more than 20%    
    assert estimated_memory_mb >= max_mem_mb 
    assert percents_relative_maxmem <= 20
    