from typing import Any, Callable, Dict, Tuple, Union
import numpy as np
import inspect
from inspect import Parameter, signature
from httomo.utils import log_once, Colour
from httomo.data import mpiutil

from mpi4py.MPI import Comm

gpu_enabled = False
try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
        gpu_enabled = True  # CuPy is installed and GPU is available
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        print("CuPy is installed but GPU device inaccessible")
except ImportError:
    import numpy as xp

    print("CuPy is not installed")


def _gpumem_cleanup():
    """cleans up GPU memory and also the FFT plan cache"""
    if gpu_enabled:
        xp.get_default_memory_pool().free_all_blocks()
        cache = xp.fft.config.get_plan_cache()
        cache.clear()


class BaseWrapper:
    """A parent class for all wrappers in httomo that use external modules."""

    def __init__(
        self, module_name: str, function_name: str, method_name: str, comm: Comm
    ):
        self.comm = comm
        self.cupyrun = False
        self.module: Any = None
        self.dict_params: Dict[str, Any] = {}
        if gpu_enabled:
            self.num_GPUs = xp.cuda.runtime.getDeviceCount()
            self.gpu_id = mpiutil.local_rank % self.num_GPUs

    def _transfer_data(self, *args) -> Union[tuple, xp.ndarray, np.ndarray]:
        """Transfer the data between the host and device for the GPU-enabled method

        Returns:
            Union[tuple, xp.ndarray, np.ndarray]: transferred datasets
        """
        if not gpu_enabled:
            return args
        xp.cuda.Device(self.gpu_id).use()
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
        reslice_ahead: bool,
    ) -> xp.ndarray:
        """The generic wrapper to execute functions for external packages.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.
            reslice_ahead (bool): a bool to inform the wrapper if the reslice ahead and the conversion to numpy required.

        Returns:
            xp.ndarray: A numpy or cupy array containing processed data.
        """
        # set the correct GPU ID if it is required
        if "gpu_id" in dict_params_method:
            dict_params_method["gpu_id"] = self.gpu_id

        # check if data needs to be transfered host <-> device
        data = self._transfer_data(data)

        data = getattr(self.module, method_name)(data, **dict_params_method)
        if reslice_ahead and gpu_enabled:
            # reslice ahead, bring data back to numpy array
            return xp.asnumpy(data)
        else:
            return data

    def _execute_normalize(
        self,
        method_name: str,
        dict_params_method: Dict,
        data: xp.ndarray,
        flats: xp.ndarray,
        darks: xp.ndarray,
        reslice_ahead: bool,
    ) -> xp.ndarray:
        """Normalisation-specific wrapper when flats and darks are required.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.
            flats (xp.ndarray): a numpy or cupy flats array.
            darks (xp.ndarray): a numpy or darks flats array.
            reslice_ahead (bool): a bool to inform the wrapper if the reslice ahead and the conversion to numpy required.

        Returns:
            xp.ndarray: a numpy or cupy array of the normalised data.
        """
        # check where data needs to be transfered host <-> device
        data, flats, darks = self._transfer_data(data, flats, darks)

        data = getattr(self.module, method_name)(
            data, flats, darks, **dict_params_method
        )
        if reslice_ahead and gpu_enabled:
            # reslice ahead, bring data back to numpy array
            return xp.asnumpy(data)
        else:
            return data

    def _execute_reconstruction(
        self,
        method_name: str,
        dict_params_method: Dict,
        data: xp.ndarray,
        angles_radians: np.ndarray,
        reslice_ahead: bool,
    ) -> xp.ndarray:
        """The reconstruction wrapper.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.
            angles_radians (np.ndarray): a numpy array of projection angles.
            reslice_ahead (bool): a bool to inform the wrapper if the reslice ahead and the conversion to numpy required.

        Returns:
            xp.ndarray: a numpy or cupy array of the reconstructed data.
        """
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
        if reslice_ahead and gpu_enabled:
            # reslice ahead, bring data back to numpy array
            return xp.asnumpy(data)
        else:
            return data

    def _execute_rotation(
        self,
        method_name: str,
        dict_params_method: Dict,
        data: xp.ndarray,
    ) -> tuple | Any:
        """The center of rotation wrapper.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            data (xp.ndarray): a numpy or cupy data array.

        Returns:
            tuple: The center of rotation and other parameters if it is 360 sinogram.
        """
        # check where data needs to be transfered host <-> device
        data = self._transfer_data(data)
        method_func = getattr(self.module, method_name)
        rot_center = 0
        overlap = 0
        side = 0
        overlap_position = 0
        mid_rank = int(round(self.comm.size / 2) + 0.1)
        if self.comm.rank == mid_rank:
            if dict_params_method["ind"] == "mid":
                dict_params_method["ind"] = data.shape[1] // 2  # get the middle slice
            if method_name == "find_center_360":
                (rot_center, overlap, side, overlap_position) = method_func(
                    data, **dict_params_method
                )
            else:
                rot_center = method_func(data, **dict_params_method)

        if method_name == "find_center_vo":
            rot_center = self.comm.bcast(rot_center, root=mid_rank)
            log_once(
                f"The center of rotation for 180 degrees sinogram is {rot_center}",
                comm=self.comm,
                colour=Colour.LYELLOW,
                level=1,
            )
            return rot_center
        if method_name == "find_center_360":
            (rot_center, overlap, side, overlap_position) = self.comm.bcast(
                (rot_center, overlap, side, overlap_position), root=mid_rank
            )
            log_once(
                f"The center of rotation for 360 degrees sinogram is {rot_center},"
                + f" overlap {overlap}, side {side} and overlap position {overlap_position}",
                self.comm,
                colour=Colour.LYELLOW,
                level=1,
            )
            return (rot_center, overlap, side, overlap_position)


class TomoPyWrapper(BaseWrapper):
    """A class that wraps TomoPy functions for httomo"""

    def __init__(
        self, module_name: str, function_name: str, method_name: str, comm: Comm
    ):
        super().__init__(module_name, function_name, method_name, comm)

        # if not changed bellow the generic wrapper will be executed
        self.wrapper_method: Callable = super()._execute_generic

        if module_name in ["misc", "prep", "recon"]:
            from importlib import import_module

            self.module = getattr(import_module("tomopy." + module_name), function_name)
            if function_name == "normalize":
                func = getattr(self.module, method_name)
                sig_params = signature(func).parameters
                if "dark" in sig_params and "flat" in sig_params:
                    self.wrapper_method = super()._execute_normalize
            if function_name == "algorithm":
                self.wrapper_method = super()._execute_reconstruction
            if function_name == "rotation":
                self.wrapper_method = super()._execute_rotation

        # As for TomoPy ver. 1.13 it is not possible to pass a CuPy array to the function
        # directly, therefore we set the flag explicitly
        self.cupyrun = False


class HttomolibWrapper(BaseWrapper):
    """A class that wraps httomolib functions for httomo"""

    def __init__(
        self, module_name: str, function_name: str, method_name: str, comm: Comm
    ):
        super().__init__(module_name, function_name, method_name, comm)

        # if not changed bellow the generic wrapper will be executed
        self.wrapper_method: Callable = super()._execute_generic

        if module_name in ["misc", "prep", "recon"]:
            from importlib import import_module

            self.module = getattr(
                import_module("httomolib." + module_name), function_name
            )
            if function_name == "images":
                self.wrapper_method = self._execute_images
            if function_name == "normalize":
                func = getattr(self.module, method_name)
                sig_params = signature(func).parameters
                if "darks" in sig_params and "flats" in sig_params:
                    self.wrapper_method = super()._execute_normalize
            if function_name == "algorithm":
                self.wrapper_method = super()._execute_reconstruction
            if function_name == "rotation":
                self.wrapper_method = super()._execute_rotation

        # httomolib exports metadata from the method decorator, which we can use to
        # check if we support CuPy
        func = getattr(self.module, method_name)
        self.meta = func.meta
        self.cupyrun = self.meta.gpu
        self.dict_params: Dict[str, Any] = {}

    def calc_max_slices(
        self,
        slice_dim: int,
        other_dims: Tuple[int, int],
        dtype: np.dtype,
        available_memory: int,
    ) -> Tuple[int, np.dtype]:
        # if the function does not support GPU, we return a very large value (as we don't
        # care about limiting slices for CPU memory for now)
        if not self.cupyrun:
            return 1000000000, dtype
        # first we need to find the default argument value from the method meta info,
        # before overriding those that are given (from YAML), for the kwargs arguments
        # to calc_max_slices
        sig: inspect.Signature = self.meta.signature
        default_args = {}
        for name, par in sig.parameters.items():
            if par.default != Parameter.empty:
                default_args[name] = par.default
        kwargs = {**default_args, **self.dict_params}
        return self.meta.calc_max_slices(
            slice_dim, other_dims, dtype, available_memory, **kwargs
        )

    def _execute_images(
        self,
        method_name: str,
        dict_params_method: Dict,
        out_dir: str,
        comm: Comm,
        data: xp.ndarray,
    ) -> None:
        """httomolib wrapper for save images function.

        Args:
            method_name (str): The name of the method to use.
            dict_params_method (Dict): A dict containing parameters of the executed method.
            out_dir (str): The output directory.
            comm (Comm): The MPI communicator.
            data (xp.ndarray): a numpy or cupy data array.

        Returns:
            None: returns None.
        """
        if gpu_enabled:
            _gpumem_cleanup()
            data = getattr(self.module, method_name)(
                xp.asnumpy(data), out_dir, comm_rank=comm.rank, **dict_params_method
            )
        else:
            data = getattr(self.module, method_name)(
                data, out_dir, comm_rank=comm.rank, **dict_params_method
            )
        return None
