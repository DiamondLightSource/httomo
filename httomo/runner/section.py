"""
Class and associated functions relating to the grouping of sequences of methods in a pipeline
into a section.
"""

import logging
from typing import Iterator, List, Optional, Tuple

from httomo.runner.output_ref import OutputRef
from httomo.runner.pipeline import Pipeline
from httomo.utils import log_once
from httomo.runner.method_wrapper import MethodWrapper

from httomo_backends.methods_database.query import Pattern


class Section:
    """
    Represents a sequence of methods in a pipeline.

    See Also
    --------
    sectionize : The rules for when a new section is introduced into a pipeline are described
        and implemented in this function.
    """

    def __init__(
        self,
        pattern: Pattern,
        max_slices: int,
        methods: List[MethodWrapper],
        is_last: bool = False,
        padding: bool = False,
    ):
        self.pattern = pattern
        self.max_slices = max_slices
        self.methods = methods
        self.is_last = is_last
        self.padding = padding

    def __iter__(self) -> Iterator[MethodWrapper]:
        return iter(self.methods)

    def __len__(self) -> int:
        return len(self.methods)

    def __getitem__(self, idx: int) -> MethodWrapper:
        return self.methods[idx]


def sectionize(pipeline: Pipeline) -> List[Section]:
    """
    Groups the methods in a pipeline into sections.

    Notes
    -----

    A new section is introduced into a pipeline under any of the following conditions:
        - the pattern changed (sino -> proj or vice versa)
        - a side output of a previous method is referenced
        - more than one padding method is included
    """

    sections: List[Section] = []

    # The functions below are internal to reduce duplication

    def is_pattern_compatible(a: Pattern, b: Pattern) -> bool:
        return a == Pattern.all or b == Pattern.all or a == b

    # loop carried variables, to build up the sections
    current_pattern: Pattern = pipeline.loader_pattern
    current_methods: List[MethodWrapper] = []
    has_padding_method: bool = False

    def references_previous_method(method: MethodWrapper) -> bool:
        # find output references in the method's parameters
        refs = [v for v in method.config_params.values() if isinstance(v, OutputRef)]
        # see if any of them reference methods in the current method list
        for r in refs:
            if r.method in current_methods:
                return True
        return False

    def is_second_padded_method(method: MethodWrapper) -> bool:
        return has_padding_method and method.padding

    def finish_section():
        sections.append(Section(current_pattern, 0, current_methods))
        if has_padding_method:
            sections[-1].padding = True

    for method in pipeline:
        if (
            not is_pattern_compatible(current_pattern, method.pattern)
            or references_previous_method(method)
            or is_second_padded_method(method)
        ):
            finish_section()
            has_padding_method = False
            if method.pattern != Pattern.all:
                current_pattern = method.pattern
            current_methods = [method]
        else:
            current_methods.append(method)
            if current_pattern == Pattern.all:
                current_pattern = method.pattern
        has_padding_method = has_padding_method or method.padding

    finish_section()
    sections[-1].is_last = True

    _backpropagate_section_patterns(pipeline, sections)
    _finalize_patterns(pipeline, sections)
    _set_method_patterns(sections)

    return sections


def _backpropagate_section_patterns(pipeline: Pipeline, sections: List[Section]):
    """Performs a backward sweep through the patterns of each section, propagating
    from the last section backwards in case the previous ones have Pattern.all.
    This makes sure the loader eventually gets the pattern that the section that follows
    has.

    Only special case: All methods have Pattern.all, which is handled separately
    """
    last_pattern = Pattern.all
    for s in reversed(sections):
        if s.pattern == Pattern.all:
            s.pattern = last_pattern
        last_pattern = s.pattern
    if pipeline.loader_pattern == Pattern.all:
        pipeline.loader_pattern = last_pattern


def _finalize_patterns(
    pipeline: Pipeline,
    sections: List[Section],
    default_pattern=Pattern.projection,
):
    # final possible ambiguity: everything is Pattern.all -> pick projection by default
    if len(sections) > 0 and sections[0].pattern == Pattern.all:
        log_once(
            "All pipeline sections support all patterns: choosing projection",
            level=logging.WARNING,
        )
        for s in sections:
            s.pattern = default_pattern
        pipeline.loader_pattern = default_pattern

    assert all(s.pattern != Pattern.all for s in sections)
    assert pipeline.loader_pattern != Pattern.all


def _set_method_patterns(sections: List[Section]):
    for s in sections:
        for m in s:
            m.pattern = s.pattern


def determine_section_padding(section: Section) -> Tuple[int, int]:
    """
    Determine the padding required for the input data to a section, based on the padding
    requirements of the methods in the section.

    Notes
    -----

    Assumes that only one method with padding will be in a section, which is consistent with
    the assumptions made by `sectionize()`.
    """
    for method in section.methods:
        if method.padding:
            return method.calculate_padding()
    return (0, 0)
