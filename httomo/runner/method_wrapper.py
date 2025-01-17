from dataclasses import dataclass
from httomo.block_interfaces import T
from httomo.utils import xp

from httomo_backends.methods_database.query import GpuMemoryRequirement, Pattern

import numpy as np
from mpi4py import MPI

import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

MethodParameterValues = Union[
    str, bool, int, float, os.PathLike, np.ndarray, xp.ndarray
]
MethodParameterDictType = Dict[
    str, Union[MethodParameterValues, List[MethodParameterValues]]
]


@dataclass
class GpuTimeInfo:
    """Captures the time spent on GPU"""

    kernel: float = 0.0
    host2device: float = 0.0
    device2host: float = 0.0


class MethodWrapper(Protocol):
    """Interface for method wrappers, that is used by the pipeline and task runners to execute
    methods in a generic way."""

    # read/write properties
    task_id: str
    pattern: Pattern

    # read-only properties
    @property
    def comm(self) -> MPI.Comm:
        """The MPI communicator used"""
        ...  # pragma: no cover

    @property
    def method(self) -> Callable:
        """The actual method underlying this wrapper"""
        ...  # pragma: no cover

    @property
    def parameters(self) -> List[str]:
        """List of parameter names of the underlying method"""
        ...  # pragma: nocover

    @property
    def memory_gpu(self) -> Optional[GpuMemoryRequirement]:
        """Memory requirements for GPU execution"""
        ...  # pragma: nocover

    @property
    def implementation(self) -> Literal["gpu", "cpu", "gpu_cupy"]:
        """Implementation of this method"""
        ...  # pragma: nocover

    @property
    def output_dims_change(self) -> bool:
        """Whether output dimensions change after executing this method"""
        ...  # pragma: nocover

    @property
    def save_result(self) -> bool:
        """Whether to save the result of this method to intermediate files"""
        ...  # pragma: nocover

    @property
    def method_name(self) -> str:
        """Returns the name of the method function"""
        ...  # pragma: nocover

    @property
    def module_path(self) -> str:
        """Returns the full module path where the method function is defined"""
        ...  # pragma: nocover

    @property
    def package_name(self) -> str:
        """The name of the top-level package where this method is implementated, e.g. 'httomolib'"""
        ...  # pragma: nocover

    @property
    def cupyrun(self) -> bool:
        """True if method runs on GPU and expects a CuPy array as inputs"""
        ...  # pragma: nocover

    @property
    def is_cpu(self) -> bool:
        """True if this is a CPU-only method"""
        ...  # pragma: nocover

    @property
    def is_gpu(self) -> bool:
        """True if this is a GPU method"""
        ...  # pragma: nocover

    @property
    def gpu_time(self) -> GpuTimeInfo:
        """Get the time spent on GPU in the last call to execute"""
        ...  # pragma: nocover

    @property
    def config_params(self) -> Dict[str, Any]:
        """Access a copy of the configuration parameters (cannot be modified directly)"""
        ...  # pragma: nocover

    @property
    def recon_algorithm(self) -> Optional[str]:
        """Determine the recon algorithm used, if the method is reconstruction.
        Otherwise return None."""
        ...  # pragma: nocover

    @property
    def padding(self) -> bool:
        """Determine if the method needs padding"""
        ...  # pragma: nocover

    @property
    def sweep(self) -> bool:
        """Determine if the method performs sweep"""
        ...  # pragma: nocover

    # Methods

    def __getitem__(self, key: str) -> MethodParameterValues:
        """Get a parameter for the method using dictionary notation (wrapper["param"])"""
        ...  # pragma: nocover

    def __setitem__(self, key: str, value: MethodParameterValues):
        """Set a parameter for the method using dictionary notation (wrapper["param"] = 42)"""
        ...  # pragma: nocover

    def append_config_params(self, params: MethodParameterDictType):
        """Appends to the configuration parameters all values that are in the given dictionary"""
        ...  # pragma: nocover

    # For a given type `T` that implements the `Block` protocol, the return type should be the
    # same `T` as the input `T` (rather than allowing the input implementor of `Block` and the
    # output implementor of `Block` to be different, which is what the signature
    # `def execute(self, block: Block) -> Block` would imply).
    def execute(self, block: T) -> T:
        """Execute the method.

        Parameters
        ----------

        block: T (implements `Block`)
            A numpy or cupy dataset, mutable (method might work in-place).

        Returns
        -------

        T (implements `Block`)
            A CPU or GPU-based dataset object with the output
        """
        ...  # pragma: nocover

    def get_side_output(self) -> Dict[str, Any]:
        """Override this method for functions that have a side output. The returned dictionary
        will be merged with the dict_params parameter passed to execute for all methods that
        follow in the pipeline"""
        ...  # pragma: nocover

    def calculate_output_dims(
        self, non_slice_dims_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate the dimensions of the output for this method"""
        ...  # pragma: nocover

    def calculate_padding(self) -> Tuple[int, int]:
        """Calculate the padding required by the method"""
        ...  # pragma: nocover

    def calculate_max_slices(
        self,
        data_dtype: np.dtype,
        non_slice_dims_shape: Tuple[int, int],
        available_memory: int,
    ) -> Tuple[int, int]:
        """If it runs on GPU, determine the maximum number of slices that can fit in the
        available memory in bytes, and return a tuple of

        (max_slices, available_memory)

        The available memory may have been adjusted for the methods that follow, in case
        something persists afterwards.
        """
        ...  # pragma: nocover
