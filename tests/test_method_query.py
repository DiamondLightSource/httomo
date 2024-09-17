from pathlib import Path
from pytest_mock import MockerFixture
import yaml
from httomo.methods_database.query import (
    YAML_DIR,
    MethodsDatabaseQuery,
    get_method_info,
)
import pytest
import numpy as np
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


def test_httomolibgpu_default_save_result():
    save_result = get_method_info(
        "httomolibgpu.prep.normalize", "normalize", "save_result_default"
    )

    assert save_result is False


def test_httomolibgpu_default_save_result_recon():
    save_result = get_method_info(
        "httomolibgpu.recon.algorithm", "FBP", "save_result_default"
    )

    assert save_result is True


def test_httomolibgpu_memory_gpu():
    memory_gpu = get_method_info(
        "httomolibgpu.prep.normalize", "normalize", "memory_gpu"
    )
    assert len(memory_gpu) == 2


def test_httomolibgpu_padding_false():
    padding = get_method_info("httomolibgpu.prep.normalize", "normalize", "padding")

    assert padding is False


def test_httomolibgpu_padding_true():
    padding = get_method_info("tomopy.misc.corr", "median_filter3d", "padding")

    assert padding is True


# this is just a quick check - until we have schema validation on the DB files
def test_all_methods_have_padding_parameter():
    # we don't care about the httomo one - easy to check
    for m in ["tomopy", "httomolib", "httomolibgpu"]:
        yaml_path = Path(YAML_DIR, f"external/{m}/{m}.yaml")
        with open(yaml_path, "r") as f:
            info = yaml.safe_load(f)
            # methods are on 3rd level
            for package_name, module in info.items():
                for f_name, file in module.items():
                    for method_name, method in file.items():
                        assert (
                            "padding" in method
                        ), f"{m}.{package_name}.{f_name}.{method_name}"
                        assert type(method["padding"]) == bool


def test_database_query_object():
    query = MethodsDatabaseQuery("httomolibgpu.prep.normalize", "normalize")
    assert query.get_pattern() == Pattern.projection
    assert query.get_output_dims_change() is False
    assert query.get_implementation() == "gpu_cupy"
    assert query.swap_dims_on_output() is False
    assert query.save_result_default() is False
    assert query.padding() is False
    mempars = query.get_memory_gpu_params()
    assert mempars.method == "module"
    assert mempars.multiplier == "None"


def test_database_query_object_recon_swap_output():
    query = MethodsDatabaseQuery("tomopy.recon.algorithm", "recon")
    assert query.swap_dims_on_output() is True


def test_database_query_calculate_memory(mocker: MockerFixture):
    class FakeModule:
        def _calc_memory_bytes_testmethod(non_slice_dims_shape, dtype, testparam):
            assert non_slice_dims_shape == (
                42,
                3,
            )
            assert dtype == np.float32
            assert testparam == 42.0
            return 10, 20

    importmock = mocker.patch(
        "httomo.methods_database.query.import_module", return_value=FakeModule
    )
    query = MethodsDatabaseQuery("sample.module.path", "testmethod")

    mem = query.calculate_memory_bytes((42, 3), np.float32, testparam=42.0)

    importmock.assert_called_with(
        "httomo.methods_database.packages.external.sample.supporting_funcs.module.path"
    )
    assert mem == (10, 20)


def test_database_query_calculate_output_dims(mocker: MockerFixture):
    class FakeModule:
        def _calc_output_dim_testmethod(non_slice_dims_shape, testparam):
            assert non_slice_dims_shape == (
                42,
                3,
            )
            assert testparam == 42.0
            return 10, 20

    importmock = mocker.patch(
        "httomo.methods_database.query.import_module", return_value=FakeModule
    )
    query = MethodsDatabaseQuery("sample.module.path", "testmethod")

    dims = query.calculate_output_dims((42, 3), testparam=42.0)

    importmock.assert_called_with(
        "httomo.methods_database.packages.external.sample.supporting_funcs.module.path"
    )
    assert dims == (10, 20)


def test_database_query_calculate_padding(mocker: MockerFixture):
    SIZE_PARAMETER = 5
    PADDING_RETURNED = (5, 5)

    class FakeModule:
        def _calc_padding_testmethod(size):
            assert size == SIZE_PARAMETER
            return PADDING_RETURNED

    importmock = mocker.patch(
        "httomo.methods_database.query.import_module", return_value=FakeModule
    )
    query = MethodsDatabaseQuery("sample.module.path", "testmethod")

    pads = query.calculate_padding(size=SIZE_PARAMETER)

    importmock.assert_called_once_with(
        "httomo.methods_database.packages.external.sample.supporting_funcs.module.path"
    )

    assert pads == PADDING_RETURNED
