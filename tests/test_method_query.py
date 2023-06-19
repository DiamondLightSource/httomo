from httomo.methods_database.query import get_httomolibgpu_method_meta, get_method_info
import pytest


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


def test_httomolibgpu_gpu_cpu():
    assert get_method_info("httomolibgpu.prep.normalize", "normalize", "gpu") is True
    assert get_method_info("httomolibgpu.prep.normalize", "normalize", "cpu") is False


def test_httomolibgpu_memfunc():
    assert callable(
        get_method_info("httomolibgpu.prep.normalize", "normalize", "calc_max_slices")
    )


def test_httomolibgpu_meta():
    from httomolibgpu import MethodMeta

    assert isinstance(get_httomolibgpu_method_meta("prep.normalize.normalize"), MethodMeta)


def test_httomolibgpu_meta_incomplete_path():
    with pytest.raises(ValueError, match="not resolving"):
        get_httomolibgpu_method_meta("prep.normalize")
