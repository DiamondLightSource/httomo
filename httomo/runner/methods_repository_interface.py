from typing import Dict, List, Literal, Protocol, Union
from typing_extensions import TypeAlias

from httomo.utils import Pattern


MemoryGpuDict: TypeAlias = Dict[
    Literal["datasets", "multipliers", "methods"], List[Union[str, int, float]]
]


class MethodQuery(Protocol):
    """An interface to query information about a single method.
    It is used by the backend wrapper classes to determine required information.

    Note: Implementers might get the information from different places, such as
    YAML files, decorators, etc, and it might also be hardcoded for specific ones.

    """

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
    """Factory method class which can obtain a query object for each method.
    It is representing a whole collection of methods and can create
    queries for each of them from various sources.
    """

    def query(self, module_path: str, method_name: str) -> MethodQuery:
        """Obtain a query object for a specifc method. Depending on the
        method given, it may create different types of MethodQuery objects"""
        ...
