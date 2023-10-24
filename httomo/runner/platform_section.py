from typing import Iterator, List

import mpi4py
from httomo.runner.pipeline import Pipeline
from httomo.utils import Colour, Pattern, log_once
from httomo.runner.backend_wrapper import BackendWrapper


class PlatformSection:
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


class PlatformSectionizer:
    def __init__(self, pipeline: Pipeline, save_all: bool = False):
        self.sections = self._determine_sections(pipeline, save_all)

    def __iter__(self) -> Iterator[PlatformSection]:
        return iter(self.sections)

    def __len__(self) -> int:
        return len(self.sections)

    def _determine_sections(
        self, pipeline: Pipeline, save_all: bool
    ) -> List[PlatformSection]:
        sections: List[PlatformSection] = []

        def should_save_after(method: BackendWrapper) -> bool:
            params = method.config_params
            return (
                save_all
                or params.get("save_result", False)
                or params.get("glob_stats", False)
            )

        def is_pattern_compatible(a: Pattern, b: Pattern) -> bool:
            return a == Pattern.all or b == Pattern.all or a == b

        itermethods = iter(pipeline.methods)
        method = next(itermethods)  # first method processed directly
        current_gpu = method.is_gpu
        current_pattern = method.pattern
        current_methods: List[BackendWrapper] = [method]
        save_result_after = should_save_after(method)

        def finish_section(needs_reslice=False):
            pattern = current_pattern
            # do the forward propagation of the pattern from loader
            if pattern == Pattern.all:
                pattern = (
                    sections[-1].pattern
                    if len(sections) > 0
                    else pipeline.loader_pattern
                )
            sections.append(
                PlatformSection(current_gpu, pattern, needs_reslice, 0, current_methods)
            )

        for i, method in enumerate(itermethods, 1):
            pattern_changed = not is_pattern_compatible(current_pattern, method.pattern)
            if save_result_after or pattern_changed or method.is_gpu != current_gpu:
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

        # we need to do a backward sweep also, in case the loader had the "all" pattern,
        last_pattern = Pattern.all
        for s in reversed(sections):
            if s.pattern == Pattern.all:
                s.pattern = last_pattern
            last_pattern = s.pattern
        if pipeline.loader_pattern == Pattern.all:
            pipeline.loader_pattern = last_pattern
        elif pipeline.loader_pattern != last_pattern:
            pipeline.loader_reslice = True 

        # final possible ambiguity: everything is Pattern.all -> pick projection by default
        if last_pattern == Pattern.all:
            log_once(
                "All pipeline sections support all patterns: choosing projection",
                mpi4py.MPI.COMM_WORLD,
                Colour.YELLOW,
                level=2,
            )
            for s in sections:
                s.pattern = Pattern.projection
            pipeline.loader_pattern = Pattern.projection

        assert all(s.pattern != Pattern.all for s in sections)
        assert pipeline.loader_pattern != Pattern.all

        return sections
