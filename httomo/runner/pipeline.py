from httomo.runner.loader import LoaderInterface
from httomo.utils import Pattern
from httomo.runner.method_wrapper import MethodWrapper

from typing import Iterator, List, Optional


class Pipeline:
    """Represents a pipeline of methods, stored by their wrappers, and the loader.
    After creation, the pipeline is immutable."""

    def __init__(
        self,
        loader: LoaderInterface,
        methods: List[MethodWrapper],
        main_pipeline_start: int = 0,
    ):
        self._methods = methods
        self._loader = loader
        self._main_pipeline_start = main_pipeline_start

    @property
    def loader(self) -> LoaderInterface:
        return self._loader

    @property
    def main_pipeline_start(self) -> int:
        return self._main_pipeline_start

    # iterator interface to access the methods
    def __iter__(self) -> Iterator[MethodWrapper]:
        return iter(self._methods)

    def __len__(self) -> int:
        return len(self._methods)

    def __getitem__(self, idx: int) -> MethodWrapper:
        return self._methods[idx]

    @property
    def loader_pattern(self) -> Pattern:
        return self.loader.pattern

    @loader_pattern.setter
    def loader_pattern(self, pattern: Pattern):
        """Although the pipeline is largely immutable, this setter is needed as the
        actual pattern is set after processing the full pipeline"""
        self.loader.pattern = pattern
