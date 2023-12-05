from typing import Any, Callable, Dict, Union
import numpy as np
from inspect import signature
from httomo.runner.gpu_utils import gpumem_cleanup
import httomo.globals
from httomo.utils import (
    Colour,
    log_exception,
    log_once,
    log_rank,
    gpu_enabled,
    xp,
)
from httomo.data import mpiutil

from mpi4py.MPI import Comm


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
        gpumem_cleanup()
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
