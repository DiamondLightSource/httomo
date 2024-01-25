from httomo.runner.dataset import DataSetBlock
from httomo.utils import xp

import numpy as np

import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

MethodParameterValues = Union[str, bool, int, float, os.PathLike, np.ndarray, xp.ndarray]
MethodParameterDictType = Dict[str, Union[MethodParameterValues, List[MethodParameterValues]]]


class MethodWrapper(Protocol):
    """Interface for method wrappers, that is used by the pipeline and task runners to execute
    methods in a generic way."""
    
    @property
    def task_id(self) -> str:
        """Returns the task id for this method"""
        ... # pragma: nocover
        
    @property
    def save_result(self) -> bool:
        """Whether to save the result of this method to intermediate files"""
        ... # pragma: nocover
    
    @property
    def method_name(self) -> str:
        """Returns the name of the method function"""
        ... # pragma: nocover
    
    @property
    def module_path(self) -> str:
        """Returns the full module path where the method function is defined"""
        ... # pragma: nocover
    
    @property
    def package_name(self) -> str:
        """The name of the top-level package where this method is implementated, e.g. 'httomolib'"""
        ... # pragma: nocover
    
    @property
    def cupyrun(self) -> bool:
        """True if method runs on GPU and expects a CuPy array as inputs"""
        ... # pragma: nocover

    @property
    def is_cpu(self) -> bool:
        """True if this is a CPU-only method"""
        ... # pragma: nocover

    @property
    def is_gpu(self) -> bool:
        """True if this is a GPU method"""
        ... # pragma: nocover
    

    def __getitem__(self, key: str) -> MethodParameterValues:
        """Get a parameter for the method using dictionary notation (wrapper["param"])"""
        ... # pragma: nocover

    def __setitem__(self, key: str, value: MethodParameterValues):
        """Set a parameter for the method using dictionary notation (wrapper["param"] = 42)"""
        ... # pragma: nocover

    @property
    def config_params(self) -> Dict[str, Any]:
        """Access a copy of the configuration parameters (cannot be modified directly)"""
        ... # pragma: nocover

    def append_config_params(self, params: MethodParameterDictType):
        """Appends to the configuration parameters all values that are in the given dictionary"""
        ... # pragma: nocover
        
    @property
    def recon_algorithm(self) -> Optional[str]:
        """Determine the recon algorithm used, if the method is reconstruction.
        Otherwise return None."""
        ... # pragma: nocover

    def execute(self, dataset: DataSetBlock) -> DataSetBlock:
        """Execute the method.

        Parameters
        ----------

        dataset: DataSetBlock
            A numpy or cupy dataset, mutable (method might work in-place).

        Returns
        -------

        DataSetBlock
            A CPU or GPU-based dataset object with the output
        """
        ... # pragma: nocover
        
    def get_side_output(self) -> Dict[str, Any]:
        """Override this method for functions that have a side output. The returned dictionary
        will be merged with the dict_params parameter passed to execute for all methods that
        follow in the pipeline"""
        ... # pragma: nocover
        
    def calculate_output_dims(
        self, non_slice_dims_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate the dimensions of the output for this method"""
        ... # pragma: nocover
        

    def calculate_max_slices(
        self,
        data_dtype: np.dtype,
        non_slice_dims_shape: Tuple[int, int],
        available_memory: int,
        darks: np.ndarray,
        flats: np.ndarray,
    ) -> Tuple[int, int]:
        """If it runs on GPU, determine the maximum number of slices that can fit in the
        available memory in bytes, and return a tuple of

        (max_slices, available_memory)

        The available memory may have been adjusted for the methods that follow, in case
        something persists afterwards.
        """
        ... # pragma: nocover
        
        





