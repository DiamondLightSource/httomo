import pytest
from pytest_mock import MockerFixture
from httomo.pipeline import Pipeline
from httomo.platform_section import PlatformSectionizer
from httomo.loader import Loader
from httomo.utils import Pattern
from httomo.wrappers_class import BackendWrapper2


def make_test_method(
    mocker: MockerFixture,
    gpu=False,
    pattern=Pattern.projection,
    method_name="testmethod",
    **kwargs,
) -> BackendWrapper2:
    mock = mocker.create_autospec(
        BackendWrapper2,
        instance=True,
        method_name=method_name,
        pattern=pattern,
        is_gpu=gpu,
        is_cpu=not gpu,
        config_params=kwargs,
        __getitem__=lambda _, k: kwargs[k],  # return kwargs value from dict access
    )

    return mock


def test_determine_single_method(mocker: MockerFixture):
    p = Pipeline()
    p.add_loader(mocker.create_autospec(Loader, instance=True, pattern=Pattern.all))
    p.append_method(make_test_method(mocker, method_name="testmethod"))
    s = PlatformSectionizer(p, False)
    assert len(s) == 1
    assert s.sections[0].methods[0].method_name == "testmethod"


def test_sectionizer_can_iterate_saveall(mocker: MockerFixture):
    p = Pipeline()
    p.add_loader(mocker.create_autospec(Loader, instance=True, pattern=Pattern.all))
    for i in range(3):
        method = make_test_method(mocker, method_name=f"testmethod{i}")
        p.append_method(method)
    s = PlatformSectionizer(p, True)
    assert len(s) == 3
    methodnames = [m.methods[0].method_name for m in s]
    assert methodnames == ["testmethod0", "testmethod1", "testmethod2"]


def test_sectionizer_two_cpu(mocker: MockerFixture):
    p = Pipeline()
    p.add_loader(mocker.create_autospec(Loader, instance=True, pattern=Pattern.all))
    p.append_method(make_test_method(mocker, pattern=Pattern.projection))
    p.append_method(make_test_method(mocker, pattern=Pattern.projection))
    s = PlatformSectionizer(p, False)
    assert len(s) == 1
    s0 = s.sections[0]
    assert len(s0) == 2
    assert s0.pattern == Pattern.projection


def test_sectionizer_pattern_change(mocker: MockerFixture):
    p = Pipeline()
    p.add_loader(mocker.create_autospec(Loader, instance=True, pattern=Pattern.all))
    p.append_method(make_test_method(mocker, pattern=Pattern.projection))
    p.append_method(make_test_method(mocker, pattern=Pattern.sinogram))
    s = PlatformSectionizer(p, False)
    assert len(s) == 2
    s0 = s.sections[0]
    assert len(s0) == 1
    assert s0.pattern == Pattern.projection
    s1 = s.sections[1]
    assert len(s1) == 1
    assert s1.pattern == Pattern.sinogram


def test_sectionizer_platform_change(mocker: MockerFixture):
    p = Pipeline()
    p.add_loader(mocker.create_autospec(Loader, instance=True, pattern=Pattern.all))
    p.append_method(make_test_method(mocker, gpu=True))
    p.append_method(make_test_method(mocker, gpu=False))
    s = PlatformSectionizer(p, False)
    assert len(s) == 2
    s0 = s.sections[0]
    assert len(s0) == 1
    assert s0.gpu is True
    s1 = s.sections[1]
    assert len(s1) == 1
    assert s1.gpu is False


@pytest.mark.parametrize(
    "loader_pattern, pattern1, pattern2, expected",
    [
        (Pattern.all, Pattern.projection, Pattern.all, Pattern.projection),
        (Pattern.all, Pattern.all, Pattern.projection, Pattern.projection),
        (Pattern.all, Pattern.sinogram, Pattern.all, Pattern.sinogram),
        (Pattern.all, Pattern.all, Pattern.sinogram, Pattern.sinogram),
        (Pattern.all, Pattern.all, Pattern.all, Pattern.projection),
    ],
    ids=[
        "proj-all-proj",
        "all-proj-proj",
        "sino-all-sino",
        "all-sino-sino",
        "all-all-all",
    ],
)
def test_determine_platform_sections_pattern_all_combine(
    mocker: MockerFixture,
    loader_pattern: Pattern,
    pattern1: Pattern,
    pattern2: Pattern,
    expected: Pattern,
):
    p = Pipeline()
    loader = mocker.create_autospec(Loader, instance=True, pattern=loader_pattern)
    p.add_loader(loader)
    p.append_method(make_test_method(mocker, pattern=pattern1))
    p.append_method(make_test_method(mocker, pattern=pattern2))

    s = PlatformSectionizer(p, False)
    assert len(s) == 1
    s0 = s.sections[0]
    assert s0.gpu is False
    assert len(s0) == 2
    assert s0.pattern == expected
    assert loader.pattern == expected


def test_sectionizer_save_result_triggers_new_section(mocker: MockerFixture):
    p = Pipeline()
    p.add_loader(mocker.create_autospec(Loader, instance=True, pattern=Pattern.all))
    p.append_method(
        make_test_method(mocker, pattern=Pattern.projection, save_result=True)
    )
    p.append_method(make_test_method(mocker, pattern=Pattern.projection))
    p.append_method(
        make_test_method(mocker, pattern=Pattern.projection, save_result=True)
    )
    p.append_method(make_test_method(mocker, pattern=Pattern.projection))

    s = PlatformSectionizer(p, False)
    assert len(s) == 3
    assert len(s.sections[0]) == 1
    assert len(s.sections[1]) == 2
    assert len(s.sections[2]) == 1


def test_sectionizer_global_stats_triggers_new_section(mocker: MockerFixture):
    p = Pipeline()
    p.add_loader(mocker.create_autospec(Loader, instance=True, pattern=Pattern.all))
    p.append_method(
        make_test_method(mocker, pattern=Pattern.projection, glob_stats=True)
    )
    p.append_method(make_test_method(mocker, pattern=Pattern.projection))
    p.append_method(
        make_test_method(mocker, pattern=Pattern.projection, glob_stats=True)
    )
    p.append_method(make_test_method(mocker, pattern=Pattern.projection))

    s = PlatformSectionizer(p, False)
    assert len(s) == 3
    assert len(s.sections[0]) == 1
    assert len(s.sections[1]) == 2
    assert len(s.sections[2]) == 1


@pytest.mark.parametrize(
    "pattern1,pattern2,needs_reslice",
    [
        (Pattern.projection, Pattern.projection, False),
        (Pattern.projection, Pattern.all, False),
        (Pattern.all, Pattern.projection, False),
        (Pattern.sinogram, Pattern.sinogram, False),
        (Pattern.sinogram, Pattern.all, False),
        (Pattern.all, Pattern.sinogram, False),
        (Pattern.projection, Pattern.sinogram, True),
        (Pattern.sinogram, Pattern.projection, True),
    ],
    ids=[
        "proj-proj",
        "proj-all",
        "all-proj",
        "sino-sino",
        "sino-all",
        "all-sino",
        "proj-sino",
        "sino-proj",
    ],
)
def test_sectionizer_needs_reslice(
    mocker: MockerFixture, pattern1: Pattern, pattern2: Pattern, needs_reslice: bool
):
    p = Pipeline()
    loader = mocker.create_autospec(Loader, instance=True, pattern=Pattern.all)
    p.add_loader(loader)
    p.append_method(make_test_method(mocker, pattern=pattern1, gpu=True))
    p.append_method(make_test_method(mocker, pattern=pattern2, gpu=False))

    s = PlatformSectionizer(p, False)
    assert len(s) == 2
    assert s.sections[0].reslice == needs_reslice
    assert s.sections[1].reslice is False
    assert loader.pattern == s.sections[0].pattern


@pytest.mark.parametrize(
    "pattern",
    [Pattern.projection, Pattern.sinogram, Pattern.all],
    ids=["proj", "sino", "all"],
)
def test_sectionizer_inherits_pattern_from_before_if_all(
    mocker: MockerFixture, pattern: Pattern
):
    p = Pipeline()
    loader = mocker.create_autospec(Loader, instance=True, pattern=Pattern.projection)
    p.add_loader(loader)
    p.append_method(make_test_method(mocker, pattern=pattern, gpu=True))
    p.append_method(make_test_method(mocker, pattern=Pattern.all, gpu=False))

    s = PlatformSectionizer(p, False)
    assert len(s) == 2
    assert s.sections[0].reslice is False
    assert s.sections[1].reslice is False
    assert (
        s.sections[1].pattern == Pattern.projection
        if pattern == Pattern.all
        else pattern
    )


@pytest.mark.parametrize("loader_pattern", [Pattern.projection, Pattern.sinogram])
def test_sectionizer_inherits_loader_pattern(
    mocker: MockerFixture, loader_pattern: Pattern
):
    p = Pipeline()
    loader = mocker.create_autospec(Loader, instance=True, pattern=loader_pattern)
    p.add_loader(loader)
    p.append_method(make_test_method(mocker, pattern=Pattern.all, gpu=True))

    s = PlatformSectionizer(p, False)
    assert len(s) == 1
    assert s.sections[0].reslice is False
    assert s.sections[0].pattern == loader_pattern


def test_sectionizer_sets_reslize_in_loader(mocker: MockerFixture):
    p = Pipeline()
    loader = mocker.create_autospec(Loader, instance=True, pattern=Pattern.sinogram)
    p.add_loader(loader)
    p.append_method(make_test_method(mocker, pattern=Pattern.projection, gpu=True))

    s = PlatformSectionizer(p, False)
    assert len(s) == 1
    assert s.sections[0].reslice is False
    assert loader.pattern == Pattern.sinogram
    assert loader.reslice is True
