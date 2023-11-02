from typing import Iterator, List

import mpi4py
from httomo.runner.pipeline import Pipeline
from httomo.utils import Colour, Pattern, log_once
from httomo.runner.backend_wrapper import BackendWrapper


class PlatformSection:
    """Represents on section of a pipeline that can be executed on the same platform,
    and has the same dataset pattern."""
    
    def __init__(
        self,
        gpu: bool,
        pattern: Pattern,
        reslice: bool,
        max_slices: int,
        methods: List[BackendWrapper],
    ):
        self.gpu = gpu
        self.pattern = pattern
        self.reslice = reslice
        self.max_slices = max_slices
        self.methods = methods

    def __iter__(self) -> Iterator[BackendWrapper]:
        return iter(self.methods)

    def __len__(self) -> int:
        return len(self.methods)


def sectionize(pipeline: Pipeline, save_all: bool = False) -> List[PlatformSection]:
    sections: List[PlatformSection] = []

    # The methods below are internal, to reduce duplication and hide them

    def should_save_after(method: BackendWrapper) -> bool:
        params = method.config_params
        return (
            save_all
            or params.get("save_result", False)
            or params.get("glob_stats", False)
        )

    def is_pattern_compatible(a: Pattern, b: Pattern) -> bool:
        return a == Pattern.all or b == Pattern.all or a == b

    itermethods = iter(pipeline)
    # first method processed directly
    method = next(itermethods)
    current_gpu = method.is_gpu
    current_pattern = method.pattern
    current_methods: List[BackendWrapper] = [method]
    save_result_after = should_save_after(method)

    def finish_section(needs_reslice=False):
        pattern = current_pattern
        # do the forward propagation of the pattern from loader
        if pattern == Pattern.all:
            pattern = (
                sections[-1].pattern if len(sections) > 0 else pipeline.loader_pattern
            )
        sections.append(
            PlatformSection(current_gpu, pattern, needs_reslice, 0, current_methods)
        )

    for _, method in enumerate(itermethods, 1):
        pattern_changed = not is_pattern_compatible(current_pattern, method.pattern)
        platform_changed = method.is_gpu != current_gpu

        if save_result_after or pattern_changed or platform_changed:
            finish_section(pattern_changed)
            current_gpu = method.is_gpu
            current_pattern = method.pattern
            current_methods = [method]
            save_result_after = should_save_after(method)
        else:
            current_methods.append(method)
            save_result_after = should_save_after(method)
            if current_pattern == Pattern.all:
                current_pattern = method.pattern

    finish_section()

    _backpropagate_section_patterns(pipeline, sections)
    _finalize_patterns(pipeline, sections)

    return sections


def _backpropagate_section_patterns(
    pipeline: Pipeline, sections: List[PlatformSection]
):
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
    elif pipeline.loader_pattern != last_pattern:
        pipeline.loader_reslice = True


def _finalize_patterns(
    pipeline: Pipeline,
    sections: List[PlatformSection],
    default_pattern=Pattern.projection,
):
    # final possible ambiguity: everything is Pattern.all -> pick projection by default
    if len(sections) > 0 and sections[0].pattern == Pattern.all:
        log_once(
            "All pipeline sections support all patterns: choosing projection",
            mpi4py.MPI.COMM_WORLD,
            Colour.YELLOW,
            level=2,
        )
        for s in sections:
            s.pattern = default_pattern
        pipeline.loader_pattern = default_pattern

    assert all(s.pattern != Pattern.all for s in sections)
    assert pipeline.loader_pattern != Pattern.all
