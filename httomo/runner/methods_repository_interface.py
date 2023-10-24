from typing import Dict, List, Literal, Protocol, Union
from typing_extensions import TypeAlias

from httomo.utils import Pattern


MemoryGpuDict: TypeAlias = Dict[
    Literal["datasets", "multipliers", "methods"], List[Union[str, int, float]]
]


class MethodsQuery(Protocol):
    def get_pattern(self) -> Pattern:
        """Return the pattern of the method"""
        ...

    def get_output_dims_change(self) -> bool:
        """Check if output dimensions change"""
        ...

    def get_implementation(self) -> Literal["cpu", "gpu", "gpu_cupy"]:
        """Check for implementation of the method"""
        ...

    def get_memory_gpu_params(self) -> MemoryGpuDict:
        """Get the parameters for the GPU memory estimation"""
        ...


class MethodRepository(Protocol):
    def query(self, module_path: str, method_name: str) -> MethodsQuery:
        ...
