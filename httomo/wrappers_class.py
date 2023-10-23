import os
from typing import Any, Callable, Dict, List, Union
import numpy as np
from abc import ABC, abstractmethod
from inspect import signature
from httomo.dataset import DataSet
import httomo.globals
from httomo.methods_query import MethodsQuery
from httomo.utils import Colour, log_exception, log_once, log_rank, gpu_enabled, xp
from httomo.data import mpiutil

from mpi4py.MPI import Comm


def _gpumem_cleanup():
    """cleans up GPU memory and also the FFT plan cache"""
    if gpu_enabled:
        xp.get_default_memory_pool().free_all_blocks()
        cache = xp.fft.config.get_plan_cache()
        cache.clear()


class MethodRepository(ABC):
    @abstractmethod
    def query(self, module_path: str, method_name: str) -> MethodsQuery:
        pass


class BackendWrapper2:
    """Defines a generic method backend wrapper in httomo which is used by task runner.

    Method parameters (configuration parameters, usually set by the user) can be set either
    using keyword arguments to the constructor, or by using conventional dictionary set/get methods
    like::

        wrapper["parameter"] = value
    """

    DictValues = Union[str, bool, int, float, os.PathLike]
    DictType = Dict[str, Union[DictValues, List[DictValues]]]

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        **kwargs,
    ):
        """Constructs a BackendWrapper for a method located in module_path with the name method_name.

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
        **kwargs:
            Additional keyword arguments will be set as configuration parameters for this
            method. Note: The method must actually accept them as parameters.
        """

        self.comm = comm
        self.module_path = module_path
        self.method_name = method_name
        from importlib import import_module

        self.module = import_module(module_path)
        self.method: Callable = getattr(self.module, method_name)
        # get all the method parameter names, so we know which to set on calling it
        sig = signature(self.method)
        self.parameters = list(sig.parameters.keys())
        # check if the kwargs are actually supported by the method
        self._config_params = kwargs
        self._check_config_params()

        # assign method properties from the methods repository
        query = method_repository.query(module_path, method_name)
        self.pattern = query.get_pattern()
        self.output_dims_change = query.get_output_dims_change()
        self.implementation = query.get_implementation()
        self.memory_gpu = query.get_memory_gpu_params()
        self.cupyrun = self.implementation == "gpu_cupy"
        self.is_cpu = self.implementation == "cpu"
        self.is_gpu = not self.is_cpu

        if gpu_enabled:
            self.num_gpus = xp.cuda.runtime.getDeviceCount()
            _id = httomo.globals.gpu_id
            self.gpu_id = mpiutil.local_rank % self.num_gpus if _id == -1 else _id

    def __getitem__(self, key: str) -> DictValues:
        """Get a parameter for the method using dictionary notation (wrapper["param"])"""
        return self._config_params[key]

    def __setitem__(self, key: str, value: DictValues):
        """Set a parameter for the method using dictionary notation (wrapper["param"] = 42)"""
        self._config_params[key] = value
        self._check_config_params()

    @property
    def config_params(self):
        """Access a copy of the configuration parameters (cannot be modified directly)"""
        return {**self._config_params}

    def append_config_params(self, params: DictType):
        """Append configuration parameters to the existing config_params"""
        self._config_params |= params
        self._check_config_params()

    def _check_config_params(self):
        for k in self._config_params.keys():
            if k not in self.parameters:
                raise ValueError(f"Unsupported keyword argument given: {k}")

    def _build_kwargs(self, dict_params: DictType, dataset: DataSet) -> Dict[str, Any]:
        # first parameter is always the data
        ret: Dict[str, Any] = dict()
        ret[self.parameters[0]] = dataset.data
        # other parameters are looked up by name
        for p in self.parameters[1:]:
            if p in dir(dataset):
                ret[p] = getattr(dataset, p)
            elif p in dict_params:
                ret[p] = dict_params[p]
            elif p == "gpu_id":
                if gpu_enabled:
                    ret[p] = self.gpu_id
                else:
                    raise ValueError(
                        f"method {self.method_name} requires gpu_id parameter, but GPU is not enabled"
                    )
            else:
                raise ValueError(f"Cannot map method parameter {p} to a value")
        return ret

    def execute(self, dataset: DataSet) -> DataSet:
        """Execute functions for external packages.

        Developer note: Derived classes may override this function or any of the methods
        it uses to modify behaviour.

        Parameters
        ----------

        dataset: DataSet
            A numpy or cupy dataset, mutable (method might work in-place).

        Returns
        -------

        DataSet
            A CPU or GPU-based dataset object with the output
        """

        dataset = self._transfer_data(dataset)
        dataset = self._preprocess_data(dataset)
        args = self._build_kwargs(self._transform_params(self._config_params), dataset)
        dataset = self._run_method(dataset, args)
        dataset = self._postprocess_data(dataset)

        return dataset

    def _run_method(self, dataset: DataSet, args: Dict[str, Any]) -> DataSet:
        """Runs the actual method - override if special handling is required
        Or side outputs are produced."""
        ret = self.method(**args)
        dataset = self._process_return_type(ret, dataset)
        return dataset

    def _process_return_type(self, ret: Any, input_dataset: DataSet) -> DataSet:
        """Checks return type of method call and assigns/creates return DataSet object.
        Override this method if a return type different from ndarray is produced and
        needs to be processed in some way.
        """
        if type(ret) != np.ndarray and type(ret) != xp.ndarray:
            raise ValueError(
                f"Invalid return type for method {self.method_name} (in module {self.module_path})"
            )
        input_dataset.data = ret
        return input_dataset

    def get_side_output(self) -> Dict[str, Any]:
        """Override this method for functions that have a side output. The returned dictionary
        will be merged with the dict_params parameter passed to execute for all methods that
        follow in the pipeline"""
        return dict()

    def _transfer_data(self, dataset: DataSet):
        if not self.cupyrun:
            return dataset
        if not gpu_enabled:
            no_gpulog_str = "GPU is not available, please use only CPU methods"
            log_once(no_gpulog_str, self.comm, colour=Colour.BVIOLET, level=1)
            return dataset

        xp.cuda.Device(self.gpu_id).use()
        gpulog_str = f"Using GPU {self.gpu_id} to transfer data of shape {xp.shape(dataset.data[0])}"
        log_rank(gpulog_str, comm=self.comm)
        _gpumem_cleanup()
        dataset.to_gpu()
        return dataset

    def _transform_params(self, dict_params: DictType) -> DictType:
        """Hook for derived classes, for transforming the names of the possible method parameters
        dictionary, for example to rename some of them or inspect them in some way"""
        return dict_params

    def _preprocess_data(self, dataset: DataSet) -> DataSet:
        """Hook for derived classes to implement proprocessing steps, after the data has been
        transferred and before the method is called"""
        return dataset

    def _postprocess_data(self, dataset: DataSet) -> DataSet:
        """Hook for derived classes to implement postprocessing steps, after the method has been
        called"""
        return dataset


class ReconstructionWrapper(BackendWrapper2):
    """Wraps reconstruction functions, limiting the length of the angles array
    before calling the method."""

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        **kwargs,
    ):
        super().__init__(method_repository, module_path, method_name, comm, **kwargs)

    def _preprocess_data(self, dataset: DataSet) -> DataSet:
        # for 360 degrees data the angular dimension will be truncated while angles are not.
        # Truncating angles if the angular dimension has got a different size
        datashape0 = dataset.data.shape[0]
        if datashape0 != len(dataset.angles_radians):
            dataset.unlock()
            dataset.angles_radians = dataset.angles_radians[0:datashape0]
            dataset.lock()
        return super()._preprocess_data(dataset)


class RotationWrapper(BackendWrapper2):
    """Wraps rotation (centering) methods, which output the original dataset untouched,
    but have a side output for the center of rotation data (handling both 180 and 360).
    """

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        **kwargs,
    ):
        super().__init__(method_repository, module_path, method_name, comm, **kwargs)
        self._side_output: Dict[str, Any] = dict()

    def _process_return_type(self, ret: Any, input_dataset: DataSet) -> DataSet:
        if type(ret) == float:
            self._side_output["cor"] = ret
        elif type(ret) == tuple:
            # cor, overlap, side, overlap_position - from find_center_360
            self._side_output["cor"] = ret[0]  # float
            self._side_output["overlap"] = ret[1]  # float
            self._side_output["side"] = ret[2]  # 0 | 1 | None
            self._side_output["overlap_position"] = ret[3]  # float

        return input_dataset

    def get_side_output(self) -> Dict[str, Any]:
        return {**super().get_side_output(), **self._side_output}


class DezingingWrapper(BackendWrapper2):
    """Wraps the remove_outlier3d method, to clean/dezing the data.
    Note that this method is applied to all elements of the dataset, i.e.
    data, darks, and flats.
    """

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        **kwargs,
    ):
        super().__init__(method_repository, module_path, method_name, comm, **kwargs)
        assert (
            method_name == "remove_outlier3d"
        ), "Only remove_outlier3d is supported at the moment"

    def execute(self, dataset: DataSet) -> DataSet:
        # check if data needs to be transfered host <-> device
        dataset = self._transfer_data(dataset)

        dataset.data = self.method(dataset.data, **self._config_params)
        dataset.unlock()
        dataset.darks = self.method(dataset.darks, **self._config_params)
        dataset.flats = self.method(dataset.flats, **self._config_params)
        dataset.lock()

        return dataset


class ImagesWrapper(BackendWrapper2):
    """Wraps image writer methods, which accept numpy (CPU) arrays as input,
    but don't actually modify the dataset. They write the information to files"""

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        out_dir: os.PathLike,
        **kwargs,
    ):
        super().__init__(method_repository, module_path, method_name, comm, **kwargs)
        self["out_dir"] = out_dir
        self["comm_rank"] = comm.rank

    # Images execute is leaving original data on the device where it is,
    # but gives the method a CPU copy of the data.
    def execute(
        self,
        dataset: DataSet,
    ) -> DataSet:
        args = self._build_kwargs(self._transform_params(self._config_params), dataset)
        if dataset.is_gpu:
            # give method a CPU copy of the data
            args[self.parameters[0]] = xp.asnumpy(dataset.data)

        self.method(**args)

        return dataset


def make_backend_wrapper(
    method_repository: MethodRepository,
    module_path: str,
    method_name: str,
    comm: Comm,
    **kwargs,
) -> BackendWrapper2:
    """Factor function to generate the appropriate wrapper based on the module
    path and method name. Clients do not need to be concerned about which particular
    derived class is returned.

    Parameters
    ----------

    method_repository: MethodRepository
        Repository of methods that we can use the query properties
    module_path: str
        Path to the module where the method is in python notation, e.g. "httomolibgpu.prep.normalize"
    method_name: str
        Name of the method (function within the given module)
    comm: Comm
        MPI communicator object
    kwargs:
        Arbitrary keyword arguments that get passed to the method as parameters.

    Returns
    -------

    BackendWrapper2
        An instance of a wrapper class
    """

    if module_path.endswith(".algorithm"):
        return ReconstructionWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            **kwargs,
        )
    if module_path.endswith(".rotation"):
        return RotationWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            **kwargs,
        )
    if method_name == "remove_outlier3d":
        return DezingingWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            **kwargs,
        )
    if module_path.endswith(".images"):
        return ImagesWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            out_dir=httomo.globals.run_out_dir,
            **kwargs,
        )
    return BackendWrapper2(
        method_repository=method_repository,
        module_path=module_path,
        method_name=method_name,
        comm=comm,
        **kwargs,
    )


class BaseWrapper:
    """A parent class for all wrappers in httomo that use external modules."""

    def __init__(
        self,
        backend_name: str,
        module_name: str,
        function_name: str,
        method_name: str,
        comm: Comm,
    ):
        self.comm = comm
        self.module: Any = None
        self.dict_params: Dict[str, Any] = {}
        if gpu_enabled:
            self.num_GPUs = xp.cuda.runtime.getDeviceCount()
            _id = httomo.globals.gpu_id
            # if gpu-id was specified in the CLI, use that
            self.gpu_id = mpiutil.local_rank % self.num_GPUs if _id == -1 else _id

    def _transfer_data(self, *args) -> Union[tuple, xp.ndarray, np.ndarray]:
        """Transfer the data between the host and device for the GPU-enabled method

        Returns:
            Union[tuple, xp.ndarray, np.ndarray]: transferred datasets
        """
        if not gpu_enabled:
            no_gpulog_str = "GPU is not available, please use only CPU methods"
            log_once(no_gpulog_str, self.comm, colour=Colour.BVIOLET, level=1)
            return args
        xp.cuda.Device(self.gpu_id).use()
        gpulog_str = (
            f"Using GPU {self.gpu_id} to transfer data of shape {xp.shape(args[0])}"
        )
        log_rank(gpulog_str, comm=self.comm)
        _gpumem_cleanup()
        if self.cupyrun:
            if len(args) == 1:
                return xp.asarray(args[0])
            else:
                return tuple(xp.asarray(d) for d in args)
        else:
            if len(args) == 1:
                return xp.asnumpy(args[0])
            else:
                return tuple(xp.asnumpy(d) for d in args)

    def _execute_generic(
        self,
        method_name: str,
        dict_params_method: Dict,
        data: xp.ndarray,
        return_numpy: bool,
        cupyrun: bool,
    ) -> xp.ndarray:
        """The generic wrapper to execute functions for external packages.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.
            return_numpy (bool): returns numpy array if set to True.
            cupyrun (bool): True if the module uses CuPy API.

        Returns:
            xp.ndarray: A numpy or cupy array containing processed data.
        """
        self.cupyrun = cupyrun
        # set the correct GPU ID if it is required
        if "gpu_id" in dict_params_method:
            dict_params_method["gpu_id"] = self.gpu_id

        # check if data needs to be transfered host <-> device
        data = self._transfer_data(data)

        data = getattr(self.module, method_name)(data, **dict_params_method)

        if cupyrun and return_numpy:
            # if data in CuPy array but we need numpy
            return data.get()  # get numpy
        elif cupyrun and not return_numpy:
            # if data in CuPy array and we need it
            return data  # return CuPy array
        else:
            return data  # return numpy

    def _execute_normalize(
        self,
        method_name: str,
        dict_params_method: Dict,
        data: xp.ndarray,
        flats: xp.ndarray,
        darks: xp.ndarray,
        return_numpy: bool,
        cupyrun: bool,
    ) -> xp.ndarray:
        """Normalisation-specific wrapper when flats and darks are required.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.
            flats (xp.ndarray): a numpy or cupy flats array.
            darks (xp.ndarray): a numpy or darks flats array.
            return_numpy (bool): returns numpy array if set to True.
            cupyrun (bool): True if the module uses CuPy API.

        Returns:
            xp.ndarray: a numpy or cupy array of the normalised data.
        """
        self.cupyrun = cupyrun
        # check where data needs to be transfered host <-> device
        data, flats, darks = self._transfer_data(data, flats, darks)

        data = getattr(self.module, method_name)(
            data, flats, darks, **dict_params_method
        )

        if cupyrun and return_numpy:
            # if data in CuPy array but we need numpy
            return data.get()  # get numpy
        elif cupyrun and not return_numpy:
            # if data in CuPy array and we need it
            return data  # return CuPy array
        else:
            return data  # return numpy

    def _execute_reconstruction(
        self,
        method_name: str,
        dict_params_method: Dict,
        data: xp.ndarray,
        angles_radians: np.ndarray,
        return_numpy: bool,
        cupyrun: bool,
    ) -> xp.ndarray:
        """The image reconstruction wrapper.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.
            angles_radians (np.ndarray): a numpy array of projection angles.
            return_numpy (bool): returns numpy array if set to True.
            cupyrun (bool): True if the module uses CuPy API.

        Returns:
            xp.ndarray: a numpy or cupy array of the reconstructed data.
        """
        self.cupyrun = cupyrun
        # set the correct GPU ID if it is required
        if "gpu_id" in dict_params_method:
            dict_params_method["gpu_id"] = self.gpu_id

        # check if data needs to be transfered host <-> device
        data = self._transfer_data(data)

        # for 360 degrees data the angular dimension will be truncated while angles are not.
        # Truncating angles if the angular dimension has got a different size
        datashape0 = data.shape[0]
        if datashape0 != len(angles_radians):
            angles_radians = angles_radians[0:datashape0]

        data = getattr(self.module, method_name)(
            data, angles_radians, **dict_params_method
        )
        if cupyrun and return_numpy:
            # if data in CuPy array but we need numpy
            return data.get()  # get numpy
        elif cupyrun and not return_numpy:
            # if data in CuPy array and we need it
            return data  # return CuPy array
        else:
            return data  # return numpy

    def _execute_rotation(
        self,
        method_name: str,
        dict_params_method: Dict,
        data: xp.ndarray,
        return_numpy: bool,
        cupyrun: bool,
    ) -> Union[tuple, Any]:
        """The center of rotation wrapper.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.
            return_numpy (bool): returns numpy array if set to True.
            cupyrun (bool): True if the module uses CuPy API.

        Returns:
            tuple: The center of rotation and other parameters if it is 360 sinogram.
        """
        self.cupyrun = cupyrun
        # check where data needs to be transfered host <-> device
        data = self._transfer_data(data)
        method_func = getattr(self.module, method_name)
        rot_center = 0
        overlap = 0
        side = 0
        overlap_position = 0

        if method_name == "find_center_360":
            (rot_center, overlap, side, overlap_position) = method_func(
                data, **dict_params_method
            )
            log_once(
                f"###___The center of rotation for 360 degrees sinogram is {rot_center},"
                + f" overlap {overlap}, side {side} and overlap position {overlap_position}___###",
                self.comm,
                colour=Colour.LYELLOW,
                level=1,
            )
            return (rot_center, overlap, side, overlap_position)
        elif method_name == "find_center_vo":
            rot_center = method_func(data, **dict_params_method)
            log_once(
                f"###____The center of rotation for 180 degrees sinogram is {rot_center}____###",
                comm=self.comm,
                colour=Colour.LYELLOW,
                level=1,
            )
            return rot_center
        else:
            err_str = f"Invalid method name: {method_name}"
            log_exception(err_str)
            raise ValueError(err_str)

    def _execute_images(
        self,
        method_name: str,
        dict_params_method: Dict,
        out_dir: str,
        comm: Comm,
        data: xp.ndarray,
        return_numpy: bool,
        cupyrun: bool,
    ) -> None:
        """Wrapper for save images function from httomolib.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            out_dir (str): The output directory.
            comm (Comm): The MPI communicator.
            data (xp.ndarray): a numpy or cupy data array.
            return_numpy (bool): returns numpy array if set to True.
            cupyrun (bool): True if the module uses CuPy API.

        Returns:
            None: returns None.
        """
        self.cupyrun = cupyrun
        # check where data needs to be transfered host <-> device
        data = self._transfer_data(data)

        if gpu_enabled:
            data = getattr(self.module, method_name)(
                xp.asnumpy(data), out_dir, comm_rank=comm.rank, **dict_params_method
            )
        else:
            data = getattr(self.module, method_name)(
                data, out_dir, comm_rank=comm.rank, **dict_params_method
            )
        return None


class BackendWrapper(BaseWrapper):
    """A class that wraps backend functions for httomo"""

    def __init__(
        self,
        backend_name: str,
        module_name: str,
        function_name: str,
        method_name: str,
        comm: Comm,
    ):
        super().__init__(backend_name, module_name, function_name, method_name, comm)

        # if not changed bellow the generic wrapper will be executed
        self.wrapper_method: Callable = super()._execute_generic

        if module_name in ["misc", "prep", "recon"]:
            from importlib import import_module

            self.module = getattr(
                import_module(backend_name + "." + module_name), function_name
            )
            # deal with special cases
            if function_name == "normalize":
                if method_name != "minus_log":
                    func = getattr(self.module, method_name)
                    sig_params = signature(func).parameters
                    if (
                        "darks"
                        or "dark" in sig_params
                        and "flats"
                        or "flat" in sig_params
                    ):
                        self.wrapper_method = super()._execute_normalize
            if function_name == "algorithm":
                self.wrapper_method = super()._execute_reconstruction
            if function_name == "rotation":
                self.wrapper_method = super()._execute_rotation
            if function_name == "images":
                self.wrapper_method = super()._execute_images
