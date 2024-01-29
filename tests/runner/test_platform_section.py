from typing import Tuple
import pytest
from pytest_mock import MockerFixture
from httomo.runner.output_ref import OutputRef
from httomo.runner.pipeline import Pipeline
from httomo.runner.platform_section import sectionize, PlatformSection
from httomo.utils import Pattern
from ..testing_utils import make_test_loader, make_test_method

# For reference, we break into new platform sections if and only if:
# - the pattern was changed (reslice)
# - an output is referenced from an earlier method (so it needs to have
#   finished with all blocks before the side output can be read)


def test_determine_single_method(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker),
        methods=[make_test_method(mocker, method_name="testmethod")],
    )
    s = sectionize(p)
    assert len(s) == 1
    assert s[0].methods[0].method_name == "testmethod"


def test_platformsection_can_iterate(mocker: MockerFixture):
    sec = PlatformSection(
        pattern=Pattern.projection,
        max_slices=0,
        methods=[
            make_test_method(mocker, method_name="test1"),
            make_test_method(mocker, method_name="test2"),
        ],
    )

    assert len(sec) == 2
    for i, m in enumerate(sec):
        assert m.method_name == f"test{i+1}"


def test_sectionizer_same_pattern(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker),
        methods=[
            make_test_method(mocker, pattern=Pattern.projection),
            make_test_method(mocker, pattern=Pattern.projection),
        ],
    )
    s = sectionize(p)
    assert len(s) == 1
    s0 = s[0]
    assert len(s0) == 2
    assert s0.pattern == Pattern.projection


def test_sectionizer_pattern_change(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker),
        methods=[
            make_test_method(mocker, pattern=Pattern.projection),
            make_test_method(mocker, pattern=Pattern.sinogram),
        ],
    )
    s = sectionize(p)
    assert len(s) == 2
    s0 = s[0]
    assert len(s0) == 1
    assert s0.pattern == Pattern.projection
    s1 = s[1]
    assert len(s1) == 1
    assert s1.pattern == Pattern.sinogram


def test_sectionizer_platform_change_has_no_effect(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker),
        methods=[
            make_test_method(mocker, gpu=True),
            make_test_method(mocker, gpu=False),
        ],
    )

    s = sectionize(p)
    assert len(s) == 1
    s0 = s[0]
    assert len(s0) == 2
    assert s0[0].is_gpu is True
    assert s0[1].is_gpu is False


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
    loader = make_test_loader(mocker, pattern=loader_pattern)
    p = Pipeline(
        loader=loader,
        methods=[
            make_test_method(mocker, pattern=pattern1),
            make_test_method(mocker, pattern=pattern2),
        ],
    )

    s = sectionize(p)
    assert len(s) == 1
    s0 = s[0]
    assert len(s0) == 2
    assert s0.pattern == expected
    assert s0[0].pattern == expected
    assert s0[1].pattern == expected
    assert loader.pattern == expected


@pytest.mark.parametrize(
    "pattern",
    [Pattern.projection, Pattern.sinogram],
    ids=["proj", "sino"],
)
def test_sectionizer_inherits_pattern_from_before_if_all(
    mocker: MockerFixture, pattern: Pattern
):
    loader = make_test_loader(mocker, pattern=Pattern.all)
    p = Pipeline(
        loader=loader,
        methods=[
            make_test_method(mocker, pattern=pattern),
            make_test_method(mocker, pattern=Pattern.all),
        ],
    )

    s = sectionize(p)
    assert len(s) == 1
    assert s[0].pattern == pattern
    assert s[0][1].pattern == pattern


@pytest.mark.parametrize("loader_pattern", [Pattern.projection, Pattern.sinogram])
def test_sectionizer_inherits_loader_pattern(
    mocker: MockerFixture, loader_pattern: Pattern
):
    p = Pipeline(
        loader=make_test_loader(mocker, pattern=loader_pattern),
        methods=[make_test_method(mocker, pattern=Pattern.all, gpu=True)],
    )

    s = sectionize(p)
    assert len(s) == 1
    assert s[0].pattern == loader_pattern


def test_sectionizer_sets_islast_single(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker, pattern=Pattern.projection),
        methods=[make_test_method(mocker, pattern=Pattern.projection)],
    )
    s = sectionize(p)

    assert s[-1].is_last is True


def test_sectionizer_sets_islast_multiple(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker, pattern=Pattern.projection),
        methods=[
            make_test_method(mocker, pattern=Pattern.projection),
            make_test_method(mocker, pattern=Pattern.sinogram),
        ],
    )
    s = sectionize(p)

    assert s[0].is_last is False
    assert s[1].is_last is True


@pytest.mark.parametrize("positions", [(0, 1), (2, 3), (0, 4), (1, 2)])
def test_sectionizer_output_ref_triggers_new_section(
    mocker: MockerFixture, positions: Tuple[int, int]
):
    referenced_method = make_test_method(mocker, method_name="referenced_method")
    referring_method = make_test_method(
        mocker,
        method_name="referring_method",
        center=OutputRef(referenced_method, "testout"),
    )
    methods = [make_test_method(mocker, method_name=f"m{i}") for i in range(5)]
    methods.insert(positions[0], referenced_method)
    methods.insert(positions[1], referring_method)
    p = Pipeline(
        loader=make_test_loader(mocker),
        methods=methods,
    )

    s = sectionize(p)
    assert len(s) == 2
    # both must be in separate sections
    assert "referenced_method" in [m.method_name for m in s[0]]
    assert "referring_method" in [m.method_name for m in s[1]]


def test_sectionizer_output_ref_after_regular_section_break_does_nothing(
    mocker: MockerFixture,
):
    referenced_method = make_test_method(
        mocker, method_name="referenced_method", patterm=Pattern.projection
    )
    referring_method = make_test_method(
        mocker,
        method_name="referring_method",
        pattern=Pattern.sinogram,
        center=OutputRef(referenced_method, "testout"),
    )
    proj = [
        make_test_method(mocker, method_name=f"p{i}", pattern=Pattern.projection)
        for i in range(5)
    ]
    proj.insert(1, referenced_method)
    sino = [
        make_test_method(mocker, method_name=f"s{i}", pattern=Pattern.sinogram)
        for i in range(5)
    ]
    sino.insert(1, referring_method)
    methods = proj + sino
    p = Pipeline(
        loader=make_test_loader(mocker),
        methods=methods,
    )

    s = sectionize(p)
    assert len(s) == 2
    # both must be in separate sections
    assert "referenced_method" in [m.method_name for m in s[0]]
    assert "referring_method" in [m.method_name for m in s[1]]
    assert len(s[0]) == 6
    assert len(s[1]) == 6

@pytest.mark.parametrize("patterns", [
    (Pattern.sinogram, Pattern.projection),
    (Pattern.projection, Pattern.sinogram)
], ids=["sino-proj", "proj-sino"])
def test_sectionizer_inserts_empty_section_if_loader_pattern_mismatches(
    mocker: MockerFixture, patterns: Tuple[Pattern, Pattern]
):
    loader = make_test_loader(mocker, pattern=patterns[0])
    p = Pipeline(
        loader=loader,
        methods=[make_test_method(mocker, pattern=patterns[1])],
    )

    s = sectionize(p)
    assert len(s) == 2
    assert loader.pattern == patterns[0]
    assert len(s[0]) == 0  # emtpy section, so that reslicing happens after loader
    assert s[0].pattern == patterns[0]
    assert len(s[1]) == 1
    assert s[1].pattern == patterns[1]
