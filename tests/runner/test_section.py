from typing import Tuple
import pytest
from pytest_mock import MockerFixture
from httomo.runner.output_ref import OutputRef
from httomo.runner.pipeline import Pipeline
from httomo.runner.section import determine_section_padding, sectionize, Section
from ..testing_utils import make_test_loader, make_test_method

from httomo_backends.methods_database.query import Pattern

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
    sec = Section(
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


@pytest.mark.parametrize(
    "patterns",
    [(Pattern.sinogram, Pattern.projection), (Pattern.projection, Pattern.sinogram)],
    ids=["sino-proj", "proj-sino"],
)
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


@pytest.mark.parametrize("padding", [False, True])
def test_sectionizer_sets_padding_property(mocker: MockerFixture, padding: bool):
    p = Pipeline(
        loader=make_test_loader(mocker, pattern=Pattern.projection),
        methods=[make_test_method(mocker, pattern=Pattern.projection, padding=padding)],
    )
    s = sectionize(p)

    assert s[-1].padding is padding


def test_sectionizer_splits_section_if_multiple_padding_methods(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker, pattern=Pattern.projection),
        methods=[
            make_test_method(mocker, pattern=Pattern.projection, padding=False),
            make_test_method(mocker, pattern=Pattern.projection, padding=True),
            make_test_method(mocker, pattern=Pattern.projection, padding=False),
            make_test_method(mocker, pattern=Pattern.projection, padding=True),
        ],
    )

    s = sectionize(p)

    assert len(s) == 2
    assert s[0].padding is True
    assert s[1].padding is True
    assert (
        len(s[0]) == 3
    )  # loader + 3 methods, as we want right before the next padding method
    assert len(s[1]) == 1  # just the last method with padding


def test_determine_section_padding_no_padding_method_in_section(
    mocker: MockerFixture,
):
    loader = make_test_loader(mocker)
    method_1 = make_test_method(mocker=mocker, padding=False)
    method_2 = make_test_method(mocker=mocker, padding=False)
    method_3 = make_test_method(mocker=mocker, padding=False)

    pipeline = Pipeline(
        loader=loader,
        methods=[method_1, method_2, method_3],
    )
    sections = sectionize(pipeline)
    section_padding = determine_section_padding(sections[0])
    assert section_padding == (0, 0)


def test_determine_section_padding_one_padding_method_only_method_in_section(
    mocker: MockerFixture,
):
    loader = make_test_loader(mocker)

    PADDING = (3, 5)
    padding_method = make_test_method(mocker=mocker, padding=True)
    mocker.patch.object(
        target=padding_method,
        attribute="calculate_padding",
        return_value=PADDING,
    )

    pipeline = Pipeline(loader=loader, methods=[padding_method])
    sections = sectionize(pipeline)
    section_padding = determine_section_padding(sections[0])
    assert section_padding == PADDING


def test_determine_section_padding_one_padding_method_and_other_methods_in_section(
    mocker: MockerFixture,
):
    loader = make_test_loader(mocker)

    PADDING = (3, 5)
    padding_method = make_test_method(mocker=mocker, padding=True)
    mocker.patch.object(
        target=padding_method,
        attribute="calculate_padding",
        return_value=PADDING,
    )
    method_1 = make_test_method(mocker=mocker, padding=False)
    method_2 = make_test_method(mocker=mocker, padding=False)
    method_3 = make_test_method(mocker=mocker, padding=False)

    pipeline = Pipeline(
        loader=loader,
        methods=[method_1, method_2, padding_method, method_3],
    )

    sections = sectionize(pipeline)
    assert len(sections[0]) == 4

    section_padding = determine_section_padding(sections[0])
    assert section_padding == PADDING
