from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Union
from typing_extensions import TypeAlias

from httomo.utils import Pattern


class MethodsQuery(ABC):
    @abstractmethod
    def get_pattern(self) -> Pattern:
        """Return the pattern of the method"""
        pass

    @abstractmethod
    def get_output_dims_change(self) -> bool:
        """Check if output dimensions change"""
        pass

    @abstractmethod
    def get_implementation(self) -> Literal["cpu", "gpu", "gpu_cupy"]:
        """Check for implementation of the method"""
        pass

    MemoryGpuDict: TypeAlias = Dict[
        Literal["datasets", "multipliers", "methods"], List[Union[str, int, float]]
    ]

    @abstractmethod
    def get_memory_gpu_params(self) -> MemoryGpuDict:
        """Get the parameters for the GPU memory estimation"""
        pass
