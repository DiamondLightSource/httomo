from httomo.methods_database.query import get_httomolib_method_meta, get_method_info
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


def test_httomolib_pattern():
    pat = get_method_info("httomolib.prep.normalize", "normalize", "pattern")
    assert pat == "projection"


def test_httomolib_gpu_cpu():
    assert get_method_info("httomolib.prep.normalize", "normalize", "gpu") is True
    assert get_method_info("httomolib.prep.normalize", "normalize", "cpu") is False


def test_httomolib_memfunc():
    assert callable(
        get_method_info("httomolib.prep.normalize", "normalize", "calc_max_slices")
    )


def test_httomlib_meta():
    from httomolib import MethodMeta

    assert isinstance(get_httomolib_method_meta("prep.normalize.normalize"), MethodMeta)


def test_httomolib_meta_incomplete_path():
    with pytest.raises(ValueError, match="not resolving"):
        get_httomolib_method_meta("prep.normalize")
