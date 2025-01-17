from importlib import import_module

import httomo.globals
from httomo.block_interfaces import T, Block
from httomo.runner.gpu_utils import get_gpu_id, gpumem_cleanup
from httomo.runner.method_wrapper import (
    GpuTimeInfo,
    MethodParameterDictType,
    MethodParameterValues,
    MethodWrapper,
)
from httomo.runner.methods_repository_interface import (
    GpuMemoryRequirement,
    MethodRepository,
)
from httomo.runner.output_ref import OutputRef
from httomo.utils import catch_gputime, catchtime, gpu_enabled, log_rank, xp


import numpy as np
from mpi4py.MPI import Comm


from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


class GenericMethodWrapper(MethodWrapper):
    """Defines a generic method backend wrapper in httomo which is used by task runner.

    Method parameters (configuration parameters, usually set by the user) can be set either
    using keyword arguments to the constructor, or by using conventional dictionary set/get methods
    like::

        wrapper["parameter"] = value
    """

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        """Method to dermine if this class should be used for wrapper instantiation,
        given the module path and method name.

        The make_method_wrapper function will iterate through all subclasses and evaluate this
        condition in the order of declaration, falling back to GenericMethodWrapper if all
        evaluate to False.

        Therefore, deriving classes should override this method to indicate the criteria
        when they should be instantiated.
        """
        return False  # pragma: no cover

    @classmethod
    def requires_preview(cls) -> bool:
        """
        Whether the wrapper class needs the preview information from the loader to execute the
        methods it wraps or not.
        """
        return False

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        save_result: Optional[bool] = None,
        output_mapping: Dict[str, str] = {},
        **kwargs,
    ):
        """Constructs a MethodWrapper for a method located in module_path with the name method_name.

        Parameters
        ----------

        method_repository: MethodRepository
            Repository that can be used to build queries about method attributes
        module_path: str
            Path to the module where the method is in python notation, e.g. "httomolibgpu.prep.normalize"
        method_name: str
            Name of the method (function within the given module)
        comm: Comm
            MPI communicator object
        save_result: Optional[bool]
            Should the method's result be saved to an intermediate h5 file? If not given (or None),
            it queries the method database for the default value.
        output_mapping: Dict[str, str]
            Mapping of side-output names to new ones. Used for propagating side outputs by name.
        **kwargs:
            Additional keyword arguments will be set as configuration parameters for this
            method. Note: The method must actually accept them as parameters.
        """

        self._comm = comm
        self._module_path = module_path
        self._method_name = method_name

        self._module = import_module(module_path)
        self._method: Callable = getattr(self._module, method_name)
        # get all the method parameter names, so we know which to set on calling it
        sig = signature(self._method)
        self._parameters = [
            k for k, p in sig.parameters.items() if p.kind != Parameter.VAR_KEYWORD
        ]
        self._params_with_defaults = [
            k for k, p in sig.parameters.items() if p.default != Parameter.empty
        ]
        self._has_kwargs = any(
            p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        self.task_id = kwargs.pop("task_id", "")

        # check if the given kwargs are actually supported by the method
        self._config_params = kwargs
        # check if a tuple present in parameters to make the method a sweep method
        self._sweep = False
        for value in self._config_params.values():
            if type(value) is tuple:
                self._sweep = True
        self._output_mapping = output_mapping
        self._check_config_params()

        # assign method properties from the methods repository
        self._query = method_repository.query(module_path, method_name)
        self.pattern = self._query.get_pattern()
        self._output_dims_change = self._query.get_output_dims_change()
        self._implementation = self._query.get_implementation()
        self._memory_gpu = self._query.get_memory_gpu_params()
        self._padding = self._query.padding()
        self._save_result = (
            self._query.save_result_default() if save_result is None else save_result
        )

        if self.is_gpu and not gpu_enabled:
            raise ValueError("GPU is not available, please use only CPU methods")

        self._side_output: Dict[str, Any] = dict()

        self._gpu_time_info = GpuTimeInfo()

        if gpu_enabled:
            _id = httomo.globals.gpu_id
            self._gpu_id = get_gpu_id() if _id == -1 else _id

    @property
    def comm(self) -> Comm:
        return self._comm

    @property
    def method(self) -> Callable:
        return self._method

    @property
    def parameters(self) -> List[str]:
        return self._parameters

    @property
    def memory_gpu(self) -> Optional[GpuMemoryRequirement]:
        return self._memory_gpu

    @property
    def implementation(self) -> Literal["gpu", "cpu", "gpu_cupy"]:
        return self._implementation

    @property
    def output_dims_change(self) -> bool:
        return self._output_dims_change

    @property
    def save_result(self) -> bool:
        return self._save_result

    @property
    def cupyrun(self) -> bool:
        return self.implementation == "gpu_cupy"

    @property
    def is_cpu(self) -> bool:
        return self.implementation == "cpu"

    @property
    def is_gpu(self) -> bool:
        return not self.is_cpu

    @property
    def gpu_time(self) -> GpuTimeInfo:
        return self._gpu_time_info

    @property
    def method_name(self) -> str:
        return self._method_name

    @property
    def module_path(self) -> str:
        return self._module_path

    @property
    def package_name(self) -> str:
        return self._module_path.split(".")[0]

    def __getitem__(self, key: str) -> MethodParameterValues:
        """Get a parameter for the method using dictionary notation (wrapper["param"])"""
        return self._config_params[key]

    def __setitem__(self, key: str, value: MethodParameterValues):
        """Set a parameter for the method using dictionary notation (wrapper["param"] = 42)"""
        self._config_params[key] = value
        self._check_config_params()

    @property
    def config_params(self) -> Dict[str, Any]:
        """Access a copy of the configuration parameters (cannot be modified directly)"""
        return {**self._config_params}

    def append_config_params(self, params: MethodParameterDictType):
        """Append configuration parameters to the existing config_params"""
        self._config_params |= params
        self._check_config_params()

    def _check_config_params(self):
        if self._has_kwargs:
            return
        for k in self._config_params.keys():
            if k not in self.parameters:
                raise ValueError(
                    f"{self._method_name}: Unsupported keyword argument given: {k}"
                )

    @property
    def recon_algorithm(self) -> Optional[str]:
        """Determine the recon algorithm used, if the method is reconstruction.
        Otherwise return None."""
        return None

    @property
    def padding(self) -> bool:
        return self._padding

    @property
    def sweep(self) -> bool:
        return self._sweep

    def _build_kwargs(
        self,
        dict_params: MethodParameterDictType,
        dataset: Optional[Block] = None,
    ) -> Dict[str, Any]:
        # first parameter is always the data (if given)
        ret: Dict[str, Any] = dict()
        startidx = 0
        if dataset is not None:
            ret[self.parameters[startidx]] = dataset.data
            startidx += 1
        # other parameters are looked up by name
        remaining_dict_params = self._resolve_output_refs(dict_params)
        for p in self.parameters[startidx:]:
            if dataset is not None and p in dir(dataset):
                ret[p] = getattr(dataset, p)
            elif p == "comm":
                ret[p] = self.comm
            elif p == "gpu_id":
                assert gpu_enabled, "methods with gpu_id parameter require GPU support"
                ret[p] = self._gpu_id
            elif (
                p == "axis"
                and p in remaining_dict_params
                and remaining_dict_params[p] == "auto"
            ):
                ret[p] = self.pattern.value
                pass
            elif p in remaining_dict_params:
                ret[p] = remaining_dict_params[p]
            elif p in self._params_with_defaults:
                pass
            else:
                raise ValueError(f"Cannot map method parameter {p} to a value")
            remaining_dict_params.pop(p, None)

        # if method supports kwargs, pass the rest as those
        if len(remaining_dict_params) > 0 and self._has_kwargs:
            ret |= remaining_dict_params
        return ret

    def _resolve_output_refs(self, dict_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for OutputRef instances and resolve their value"""
        ret: Dict[str, Any] = dict()
        for k, v in dict_params.items():
            if isinstance(v, OutputRef):
                v = v.value
            ret[k] = v
        return ret

    def execute(self, block: T) -> T:
        """Execute functions for external packages.

        Developer note: Derived classes may override this function or any of the methods
        it uses to modify behaviour.

        Parameters
        ----------

        block: T (implements `Block`)
            A numpy or cupy dataset, mutable (method might work in-place).

        Returns
        -------

        T (implements `Block`)
            A CPU or GPU-based dataset object with the output
        """

        self._gpu_time_info = GpuTimeInfo()
        block = self._transfer_data(block)
        with catch_gputime() as t:
            block = self._preprocess_data(block)
            args = self._build_kwargs(
                self._transform_params(self._config_params), block
            )
            block = self._run_method(block, args)
            block = self._postprocess_data(block)

        self._gpu_time_info.kernel = t.elapsed

        return block

    def _run_method(self, block: T, args: Dict[str, Any]) -> T:
        """Runs the actual method - override if special handling is required
        Or side outputs are produced."""
        ret = self._method(**args)
        block = self._process_return_type(ret, block)
        return block

    def _process_return_type(self, ret: Any, input_block: T) -> T:
        """Checks return type of method call and assigns/creates a `T` object that
        implements `Block` (the same type `T` that was given as input). Override this method if
        a return type different from ndarray is produced and needs to be processed in some way.
        """
        if type(ret) != np.ndarray and type(ret) != xp.ndarray:
            raise ValueError(
                f"Invalid return type for method {self._method_name} (in module {self._module_path})"
            )
        if self._query.swap_dims_on_output():
            ret = ret.swapaxes(0, 1)
        input_block.data = ret
        return input_block

    def get_side_output(self) -> Dict[str, Any]:
        """Override this method for functions that have a side output. The returned dictionary
        will be merged with the dict_params parameter passed to execute for all methods that
        follow in the pipeline"""
        return {v: self._side_output[k] for k, v in self._output_mapping.items()}

    def _transfer_data(self, block: T) -> T:
        if not self.cupyrun:
            with catchtime() as t:
                block.to_cpu()
            self._gpu_time_info.device2host = t.elapsed
            return block

        assert gpu_enabled, "GPU method used on a system without GPU support"

        xp.cuda.Device(self._gpu_id).use()
        gpulog_str = (
            f"Using GPU {self._gpu_id} to transfer data of shape {xp.shape(block.data)}, "
            f"{self.method_name} ({self.package_name})"
        )
        log_rank(gpulog_str, comm=self.comm)
        gpumem_cleanup()
        with catchtime() as t:
            block.to_gpu()
        self._gpu_time_info.host2device = t.elapsed
        return block

    def _transform_params(
        self, dict_params: MethodParameterDictType
    ) -> MethodParameterDictType:
        """Hook for derived classes, for transforming the names of the possible method parameters
        dictionary, for example to rename some of them or inspect them in some way"""
        return dict_params

    def _preprocess_data(self, block: T) -> T:
        """Hook for derived classes to implement proprocessing steps, after the data has been
        transferred and before the method is called"""
        return block

    def _postprocess_data(self, block: T) -> T:
        """Hook for derived classes to implement postprocessing steps, after the method has been
        called"""
        return block

    def calculate_output_dims(
        self, non_slice_dims_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate the dimensions of the output for this method"""
        if self.output_dims_change:
            return self._query.calculate_output_dims(
                non_slice_dims_shape, **self._unwrap_output_ref_values()
            )

        return non_slice_dims_shape

    def calculate_padding(self) -> Tuple[int, int]:
        """Calculate the padding required by the method"""
        if self.padding:
            return self._query.calculate_padding(**self._unwrap_output_ref_values())
        return (0, 0)

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
        if self.is_cpu or not gpu_enabled:
            return int(100e9), available_memory

        # if we have no information, we assume in-place operation with no extra memory
        if self.memory_gpu is None:
            return (
                int(
                    available_memory
                    // (np.prod(non_slice_dims_shape) * data_dtype.itemsize)
                ),
                available_memory,
            )

        # NOTE: This could go directly into the methodquery / method database,
        # and here we just call calculated_memory_bytes
        memory_bytes_method = 0
        subtract_bytes = 0
        if self.memory_gpu.method == "direct":
            assert self.memory_gpu.multiplier is not None
            # this calculation assumes a direct (simple) correspondence through multiplier
            memory_bytes_method += int(
                self.memory_gpu.multiplier
                * np.prod(non_slice_dims_shape)
                * data_dtype.itemsize
            )
        else:
            (
                memory_bytes_method,
                subtract_bytes,
            ) = self._query.calculate_memory_bytes(
                non_slice_dims_shape, data_dtype, **self._unwrap_output_ref_values()
            )

        if memory_bytes_method == 0:
            return available_memory - subtract_bytes, available_memory

        return (
            available_memory - subtract_bytes
        ) // memory_bytes_method, available_memory

    def _unwrap_output_ref_values(self) -> Dict[str, Any]:
        """
        Iterate through params in `self.config_params` and, for any value of type `OutputRef`,
        extract the value inside the `OutputRef`.

        Returns
        -------
        Dict[str, Any]
            A dict containing all parameters in `self.config_params`, but with any `OutputRef`
            values replaced with the value inside the `OutputRef`.
        """
        params = dict()
        for name, value in self.config_params.items():
            if isinstance(value, OutputRef):
                params[name] = value.value
                continue
            params[name] = value
        return params
