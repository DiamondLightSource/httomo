import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol, Tuple, Union

from httomo_backends.methods_database.query import GpuMemoryRequirement, Pattern


class MethodQuery(Protocol):
    """An interface to query information about a single method.
    It is used by the backend wrapper classes to determine required information.

    Note: Implementers might get the information from different places, such as
    YAML files, decorators, etc, and it might also be hardcoded for specific ones.

    """

    def get_pattern(self) -> Pattern:
        """Return the pattern of the method"""
        ...  # pragma: no cover

    def get_output_dims_change(self) -> bool:
        """Check if output dimensions change"""
        ...  # pragma: no cover

    def get_implementation(self) -> Literal["cpu", "gpu", "gpu_cupy"]:
        """Check for implementation of the method"""
        ...  # pragma: no cover

    def get_memory_gpu_params(self) -> Optional[GpuMemoryRequirement]:
        """Get the parameters for the GPU memory estimation"""
        ...  # pragma: no cover

    def save_result_default(self) -> bool:
        """Check if this method saves results by default"""
        ...  # pragma: no cover

    def swap_dims_on_output(self) -> bool:
        """Check if the output 3D array needs to wap axis 0 and 1 to match httomolib.
        (This is typically true for tomopy recon methods)"""
        ...  # pragma: no cover

    def padding(self) -> bool:
        """Check if the method requires padding (i.e. is 3D and requires overlap
        regions in slicing dimension)"""
        ...  # pragma: no cover

    def calculate_memory_bytes(
        self, non_slice_dims_shape: Tuple[int, int], dtype: np.dtype, **kwargs
    ) -> Tuple[int, int]:
        """Calculate the memory required in bytes, returning bytes method and subtract bytes tuple"""
        ...  # pragma: no cover

    def calculate_output_dims(
        self, non_slice_dims_shape: Tuple[int, int], **kwargs
    ) -> Tuple[int, int]:
        """Calculate size of the non-slice dimensions for this method"""
        ...  # pragma: no cover

    def calculate_padding(self, **kwargs) -> Tuple[int, int]:
        """Calculate how much padding is needed for the method, before and after the core,
        in number of slices"""
        ...  # pragma: no cover


class MethodRepository(Protocol):
    """Factory method class which can obtain a query object for each method.
    It is representing a whole collection of methods and can create
    queries for each of them from various sources.
    """

    def query(self, module_path: str, method_name: str) -> MethodQuery:
        """Obtain a query object for a specifc method. Depending on the
        method given, it may create different types of MethodQuery objects"""
        ...  # pragma: no cover
