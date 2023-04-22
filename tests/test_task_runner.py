from unittest import mock
import pytest
import numpy as np

from httomo.task_runner import (
    MethodFunc,
    PlatformSection,
    _update_max_slices,
    _determine_platform_sections,
)
from httomo.utils import Pattern


def _dummy():
    pass


def make_test_method(
    gpu=False,
    is_loader=False,
    pattern=Pattern.projection,
    module_name="testmodule",
    wrapper_function=None,
    calc_max_slices=None,
):
    return MethodFunc(
        cpu=not gpu,
        gpu=gpu,
        is_loader=is_loader,
        module_name=module_name,
        pattern=pattern,
        method_function=_dummy,
        wrapper_function=wrapper_function,
        calc_max_slices=calc_max_slices,
    )


def test_determine_platform_sections_single() -> None:
    methods = [make_test_method(is_loader=True, module_name="testloader")]
    sections = _determine_platform_sections(methods)

    assert len(sections) == 1
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == methods
    assert s0.pattern == methods[0].pattern


def test_determine_platform_sections_two_cpu() -> None:
    methods = [
        make_test_method(is_loader=True, module_name="testloader"),
        make_test_method(),
    ]
    sections = _determine_platform_sections(methods)

    assert len(sections) == 1
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == methods
    assert s0.pattern == methods[0].pattern


def test_determine_platform_sections_pattern_change() -> None:
    methods = [
        make_test_method(
            is_loader=True, module_name="testloader", pattern=Pattern.projection
        ),
        make_test_method(pattern=Pattern.sinogram),
    ]
    sections = _determine_platform_sections(methods)

    assert len(sections) == 2
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == [methods[0]]
    assert s0.pattern == methods[0].pattern
    s1 = sections[1]
    assert s1.gpu is False
    assert s1.methods == [methods[1]]
    assert s1.pattern == methods[1].pattern


def test_determine_platform_sections_platform_change() -> None:
    methods = [
        make_test_method(is_loader=True, module_name="testloader"),
        make_test_method(gpu=True),
    ]
    sections = _determine_platform_sections(methods)

    assert len(sections) == 2
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == [methods[0]]
    assert s0.pattern == methods[0].pattern
    s1 = sections[1]
    assert s1.gpu is True
    assert s1.methods == [methods[1]]
    assert s1.pattern == methods[1].pattern


@pytest.mark.parametrize(
    "pattern1, pattern2, expected",
    [
        (Pattern.projection, Pattern.all, Pattern.projection),
        (Pattern.all, Pattern.projection, Pattern.projection),
        (Pattern.sinogram, Pattern.all, Pattern.sinogram),
        (Pattern.all, Pattern.sinogram, Pattern.sinogram),
        (Pattern.all, Pattern.all, Pattern.all),
    ],
)
def test_determine_platform_sections_pattern_all_combine(
    pattern1: Pattern, pattern2: Pattern, expected: Pattern
) -> None:
    methods = [
        make_test_method(pattern=pattern1, is_loader=True, module_name="testloader"),
        make_test_method(pattern=pattern2),
    ]
    sections = _determine_platform_sections(methods)

    assert len(sections) == 1
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == methods
    assert s0.pattern == expected


def test_platform_section_max_slices():
    max_slices_20 = mock.Mock(return_value=(20, np.float32()))
    max_slices_50 = mock.Mock(return_value=(50, np.float32()))
    max_slices_30 = mock.Mock(return_value=(30, np.float32()))
    section = PlatformSection(
        gpu=True,
        pattern=Pattern.projection,
        max_slices=0,
        methods=[
            make_test_method(
                pattern=Pattern.projection, gpu=True, calc_max_slices=max_slices_20
            ),
            make_test_method(
                pattern=Pattern.projection, gpu=True, calc_max_slices=max_slices_50
            ),
            make_test_method(
                pattern=Pattern.projection, gpu=True, calc_max_slices=max_slices_30
            ),
            make_test_method(
                pattern=Pattern.projection, gpu=True, calc_max_slices=None
            ),
        ],
    )
    with mock.patch(
        "httomo.task_runner._get_available_gpu_memory", return_value=100000
    ):
        dtype = _update_max_slices(section, (1000, 24, 42), np.uint8())

    assert section.max_slices == 20
    assert dtype == np.float32()
    # this also checks if the data type is respected - we give uint8 as input,
    # it returns float32, and the subsequent methods get the float32 as input
    max_slices_20.assert_called_once_with(0, (24, 42), np.uint8(), 100000)
    max_slices_30.assert_called_once_with(0, (24, 42), np.float32(), 100000)
    max_slices_50.assert_called_once_with(0, (24, 42), np.float32(), 100000)
