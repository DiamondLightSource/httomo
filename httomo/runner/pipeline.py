from httomo.runner.loader import LoaderInterface
from httomo.utils import Pattern
from httomo.runner.backend_wrapper import BackendWrapper

from typing import Iterator, List, Optional


class Pipeline:
    """Represents a pipeline of methods, stored by their wrappers, and the loader.
    After creation, the pipeline is immutable."""

    def __init__(self, loader: LoaderInterface, methods: List[BackendWrapper]):
        self._methods = methods
        self._loader = loader

    @property
    def loader(self) -> LoaderInterface:
        return self._loader

    # iterator interface to access the methods
    def __iter__(self) -> Iterator[BackendWrapper]:
        return iter(self._methods)

    def __len__(self) -> int:
        return len(self._methods)

    @property
    def loader_pattern(self) -> Pattern:
        if self.loader is not None:
            return self.loader.pattern
        else:
            raise ValueError("Attempt to get loader pattern, but no loader has be set")

    @loader_pattern.setter
    def loader_pattern(self, pattern: Pattern):
        """Although the pipeline is largely immutable, this setter is needed as the
        actual pattern is set after processing the full pipeline"""
        if self.loader is not None:
            self.loader.pattern = pattern
        else:
            raise ValueError("Attempt to set loader pattern, but no loader has be set")

    @property
    def loader_reslice(self) -> bool:
        return self.loader.reslice if self.loader is not None else False

    @loader_reslice.setter
    def loader_reslice(self, reslice: bool):
        """Although the pipeline is largely immutable, this setter is needed as the
        information whether reslicing is required after the loader is set later"""
        if self.loader is not None:
            self.loader.reslice = reslice
        else:
            raise ValueError(
                "Attempt to set loader reslice property, but no loader has be set"
            )
