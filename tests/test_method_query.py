from httomo.methods_database.query import MethodsDatabaseQuery, get_method_info
import pytest

from httomo.utils import Pattern


def test_get_from_tomopy():
    pat = get_method_info("tomopy.misc.corr", "median_filter", "pattern")
    assert pat == "all"


def test_get_invalid_package():
    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        get_method_info("unavailable.misc.corr", "median_filter", "pattern")


def test_get_invalid_module():
    with pytest.raises(KeyError, match="key doesntexist is not present"):
        get_method_info("tomopy.doesntexist.corr", "median_filter", "pattern")


def test_get_invalid_method():
    with pytest.raises(KeyError, match="key doesntexist is not present"):
        get_method_info("tomopy.misc.corr", "doesntexist", "pattern")


def test_get_invalid_attr():
    with pytest.raises(KeyError, match="attribute doesntexist is not present"):
        get_method_info("tomopy.misc.corr", "median_filter", "doesntexist")


def test_httomolibgpu_pattern():
    pat = get_method_info("httomolibgpu.prep.normalize", "normalize", "pattern")
    assert pat == "projection"


def test_httomolibgpu_implementation():
    implementation = get_method_info(
        "httomolibgpu.prep.normalize", "normalize", "implementation"
    )
    assert implementation == "gpu_cupy"


def test_httomolibgpu_output_dims_change():
    output_dims_change = get_method_info(
        "httomolibgpu.prep.normalize", "normalize", "output_dims_change"
    )
    assert output_dims_change == False


def test_httomolibgpu_memory_gpu():
    memory_gpu = get_method_info(
        "httomolibgpu.prep.normalize", "normalize", "memory_gpu"
    )
    assert len(memory_gpu) == 3


def test_database_query_object():
    query = MethodsDatabaseQuery("httomolibgpu.prep.normalize", "normalize")
    assert query.get_pattern() == Pattern.projection
    assert query.get_output_dims_change() is False
    assert query.get_implementation() == "gpu_cupy"
    mempars = query.get_memory_gpu_params()
    assert len(mempars) == 3
    assert set(p.dataset for p in mempars) == set(["tomo", "flats", "darks"])
    assert all(p.method == "direct" for p in mempars)
    assert all(p.multiplier >= 1.0 for p in mempars)