import pytest

from httomo.task_runner import MethodFunc, _determine_gpu_sections
from httomo.utils import Pattern


def _dummy():
    pass

def make_test_method(gpu=False, is_loader=False, pattern=Pattern.projection, module_name='testmodule'):
    return MethodFunc(
            cpu=not gpu,
            gpu=gpu,
            is_loader=is_loader,
            module_name=module_name,
            pattern=pattern,
            method_function=_dummy
        )


def test_determine_gpu_sections_single() -> None:
    methods = [ make_test_method(is_loader=True, module_name="testloader") ]
    sections = _determine_gpu_sections(methods)

    assert len(sections) == 1
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == methods
    assert s0.pattern == methods[0].pattern


def test_determine_platform_sections_two_cpu() -> None:
    methods = [
        make_test_method(is_loader=True, module_name="testloader"),
        make_test_method()
    ]
    sections = _determine_gpu_sections(methods)

    assert len(sections) == 1
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == methods
    assert s0.pattern == methods[0].pattern


def test_determine_platform_sections_pattern_change() -> None:
    methods = [
        make_test_method(is_loader=True, module_name="testloader", pattern=Pattern.projection),
        make_test_method(pattern=Pattern.sinogram)
    ]
    sections = _determine_gpu_sections(methods)

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
        make_test_method(gpu=True)
    ]
    sections = _determine_gpu_sections(methods)

    assert len(sections) == 2
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == [methods[0]]
    assert s0.pattern == methods[0].pattern
    s1 = sections[1]
    assert s1.gpu is True
    assert s1.methods == [methods[1]]
    assert s1.pattern == methods[1].pattern
    
@pytest.mark.parametrize("pattern1, pattern2, expected", [
    (Pattern.projection, Pattern.all, Pattern.projection),
    (Pattern.all, Pattern.projection, Pattern.projection),
    (Pattern.sinogram, Pattern.all, Pattern.sinogram),
    (Pattern.all, Pattern.sinogram, Pattern.sinogram),
    (Pattern.all, Pattern.all, Pattern.all)
])
def test_determine_platform_sections_pattern_all_combine(pattern1: Pattern,
                                                         pattern2: Pattern,
                                                         expected: Pattern) -> None:
    methods = [
        make_test_method(pattern=pattern1, is_loader=True, module_name="testloader"),
        make_test_method(pattern=pattern2)
    ]
    sections = _determine_gpu_sections(methods)

    assert len(sections) == 1
    s0 = sections[0]
    assert s0.gpu is False
    assert s0.methods == methods
    assert s0.pattern == expected
    