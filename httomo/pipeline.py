from httomo.loader import Loader
from httomo.utils import Pattern
from httomo.wrappers_class import BackendWrapper2

from typing import List, Optional


class Pipeline:
    """Represents a pipeline of methods, stored by their wrappers, and the loader"""

    def __init__(self):
        self.methods: List[BackendWrapper2] = []
        self.loader: Optional[Loader] = None

    def add_loader(self, loader: Loader):
        self.loader = loader

    @property
    def loader_pattern(self) -> Pattern:
        if self.loader is not None:
            return self.loader.pattern
        else:
            raise ValueError("Attempt to get loader pattern, but no loader has be set")

    @loader_pattern.setter
    def loader_pattern(self, pattern: Pattern):
        if self.loader is not None:
            self.loader.pattern = pattern
        else:
            raise ValueError("Attempt to set loader pattern, but no loader has be set")

    @property
    def loader_reslice(self) -> bool:
        return self.loader.reslice if self.loader is not None else False

    @loader_reslice.setter
    def loader_reslice(self, reslice: bool):
        if self.loader is not None:
            self.loader.reslice = reslice
        else:
            raise ValueError(
                "Attempt to set loader reslice property, but no loader has be set"
            )

    def append_method(self, method: BackendWrapper2):
        self.methods.append(method)
