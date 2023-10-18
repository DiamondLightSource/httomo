import os
from typing import Any, Callable, Dict, List, Union
import numpy as np
from inspect import signature
from httomo.dataset import DataSet
import httomo.globals
from httomo.utils import Colour, log_exception, log_once, log_rank, gpu_enabled, xp
from httomo.data import mpiutil

from mpi4py.MPI import Comm


def _gpumem_cleanup():
    """cleans up GPU memory and also the FFT plan cache"""
    if gpu_enabled:
        xp.get_default_memory_pool().free_all_blocks()
        cache = xp.fft.config.get_plan_cache()
        cache.clear()


class BackendWrapper2:
    """Defines a generic method backend wrapper in httomo which is used by task runner."""

    DictValues = Union[str, bool, int, float, os.PathLike]
    DictType = Dict[str, Union[DictValues, List[DictValues]]]

    def __init__(self, module_path: str, method_name: str, comm: Comm, cupyrun: bool):
        """Constructs a BackendWrapper for a method located in module_path with the name method_name.

        Parameters
        ----------

        module_path: str
            Path to the module where the method is in python notation, e.g. "httomolibgpu.prep.normalize"
        method_name: str
            Name of the method (function within the given module)
        comm: Comm
            MPI communicator object
        cupyrun: bool
            whether the method supports running with CuPy array inputs
        """

        self.comm = comm
        self.module_path = module_path
        self.method_name = method_name
        self.cupyrun = cupyrun
        from importlib import import_module

        self.module = import_module(module_path)
        self.method: Callable = getattr(self.module, method_name)
        # get all the method parameter names, so we know which to set on calling it
        sig = signature(self.method)
        self.parameters = list(sig.parameters.keys())
        if gpu_enabled:
            self.num_gpus = xp.cuda.runtime.getDeviceCount()
            _id = httomo.globals.gpu_id
            self.gpu_id = mpiutil.local_rank % self.num_gpus if _id == -1 else _id

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
            else:
                raise ValueError(f"Cannot map method parameter {p} to a value")
        return ret

    def execute(
        self, dict_params: DictType, dataset: DataSet, return_numpy: bool
    ) -> DataSet:
        """Execute functions for external packages.

        Developer note: Derived classes may override this function or any of the methods
        it uses to modify behaviour.

        Parameters
        ----------

        dict_params: DictType
            A dict containing parameters of the executed method, e.g. from Yaml file
        dataset: DataSet
            A numpy or cupy dataset, mutable (method might work in-place).
        return_numpy: bool
            Returns numpy array if set to True (use if followed by a CPU method).

        Returns
        -------

        DataSet
            A CPU or GPU-based dataset object with the output
        """

        # set the correct GPU ID if it is required
        if "gpu_id" in dict_params:
            dict_params["gpu_id"] = self.gpu_id

        dataset = self._transfer_data(dataset)
        dataset = self._preprocess_data(dataset)
        args = self._build_kwargs(self._transform_params(dict_params), dataset)
        dataset = self._run_method(dataset, args)
        dataset = self._postprocess_data(dataset)
        if return_numpy:
            dataset.to_cpu()

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

    def __init__(self, module_path: str, method_name: str, comm: Comm, cupyrun: bool):
        super().__init__(module_path, method_name, comm, cupyrun)

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

    def __init__(self, module_path: str, method_name: str, comm: Comm, cupyrun: bool):
        super().__init__(module_path, method_name, comm, cupyrun)
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

    def __init__(self, module_path: str, method_name: str, comm: Comm, cupyrun: bool):
        super().__init__(module_path, method_name, comm, cupyrun)
        assert (
            method_name == "remove_outlier3d"
        ), "Only remove_outlier3d is supported at the moment"

    def execute(
        self,
        dict_params: BackendWrapper2.DictType,
        dataset: DataSet,
        return_numpy: bool,
    ) -> DataSet:
        # check if data needs to be transfered host <-> device
        dataset = self._transfer_data(dataset)

        args: Dict[str, Any] = dict()
        if "kernel_size" in dict_params:
            args["kernel_size"] = dict_params["kernel_size"]
        if "dif" in dict_params:
            args["dif"] = dict_params["dif"]
        dataset.data = self.method(dataset.data, **args)
        dataset.unlock()
        dataset.darks = self.method(dataset.darks, **args)
        dataset.flats = self.method(dataset.flats, **args)
        dataset.lock()

        if return_numpy:
            dataset.to_cpu()

        return dataset


class ImagesWrapper(BackendWrapper2):
    """Wraps image writer methods, which accept numpy (CPU) arrays as input,
    but don't actually modify the dataset. They write the information to files"""

    def __init__(
        self,
        module_path: str,
        method_name: str,
        comm: Comm,
        out_dir: os.PathLike,
    ):
        super().__init__(module_path, method_name, comm, True)
        self.out_dir = out_dir

    # Images execute is leaving original data on the device where it is,
    # but gives the method a CPU copy of the data.
    def execute(
        self,
        dict_params: BackendWrapper2.DictType,
        dataset: DataSet,
        return_numpy: bool,
    ) -> DataSet:
        # if user wants numpy results, we transfer first (images methods are assumed to work on
        # numpy data anyway, not on GPU data)
        if return_numpy:
            dataset.to_cpu()

        args = self._build_kwargs(self._transform_params(dict_params), dataset)
        if dataset.is_gpu:
            # give method a CPU copy of the data
            args[self.parameters[0]] = xp.asnumpy(dataset.data)

        self.method(**args)

        return dataset

    def _transform_params(
        self, dict_params: BackendWrapper2.DictType
    ) -> BackendWrapper2.DictType:
        return {**dict_params, "out_dir": self.out_dir, "comm_rank": self.comm.rank}


def make_backend_wrapper(
    module_path: str, method_name: str, comm: Comm, cupyrun: bool = True
) -> BackendWrapper2:
    """Factor function to generate the appropriate wrapper based on the module
    path and method name. Clients do not need to be concerned about which particular
    derived class is returned.

    Parameters
    ----------

    module_path: str
        Path to the module where the method is in python notation, e.g. "httomolibgpu.prep.normalize"
    method_name: str
        Name of the method (function within the given module)
    comm: Comm
        MPI communicator object
    cupyrun: bool
        whether the method supports running with CuPy array inputs

    Returns
    -------

    BackendWrapper2
        An instance of a wrapper class
    """

    if module_path.endswith(".algorithm"):
        return ReconstructionWrapper(
            module_path=module_path, method_name=method_name, comm=comm, cupyrun=cupyrun
        )
    if module_path.endswith(".rotation"):
        return RotationWrapper(
            module_path=module_path, method_name=method_name, comm=comm, cupyrun=cupyrun
        )
    if method_name == "remove_outlier3d":
        return DezingingWrapper(
            module_path=module_path, method_name=method_name, comm=comm, cupyrun=cupyrun
        )
    if module_path.endswith(".images"):
        return ImagesWrapper(
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            out_dir=httomo.globals.run_out_dir,
        )
    return BackendWrapper2(
        module_path=module_path, method_name=method_name, comm=comm, cupyrun=cupyrun
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
