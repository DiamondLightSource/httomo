from typing import Dict
import numpy as np
import inspect
from mpi4py.MPI import Comm
from inspect import signature
from httomo.utils import print_once


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


class BaseWrapper:
    """A parent class for all wrappers in httomo that use external modules."""

    def __init__(
        self, module_name: str, function_name: str, method_name: str, comm: Comm
    ):
        self.comm = comm
        if gpu_enabled:
            self.num_GPUs = xp.cuda.runtime.getDeviceCount()
            self.gpu_id = int(comm.rank / comm.size * self.num_GPUs)
            xp._default_memory_pool.free_all_blocks()
            xp.cuda.Device(self.gpu_id).use()  # use a particular GPU by its id

    def _execute_generic(
        self, method_name: str, params: Dict, data: xp.ndarray
    ) -> xp.ndarray:
        """The generic wrapper to execute functions for external packages.

        Args:
            method_name (str): The name of the method to use.
            params (Dict): A dict containing independent of httomo params.
            data (xp.ndarray): a numpy or cupy data array.

        Returns:
            xp.ndarray: A numpy or cupy array containing processed data.
        """
        # TODO: deal with ncore more accurately
        params.pop("ncore", None)  #: silently remove `ncore` param.

        if gpu_enabled:
            if self.cupyrun:
                # the method accepts CuPy arrays for the GPU processing
                with xp.cuda.Device(self.gpu_id):
                    # move the data to a particular device
                    data = xp.asarray(data)
            else:
                # the method doesn't accept CuPy arrays
                data = xp.asnumpy(data)

        data = getattr(self.module, method_name)(data, **params)
        return data

    def _execute_normalize(
        self,
        method_name: str,
        params: Dict,
        data: xp.ndarray,
        flats: xp.ndarray,
        darks: xp.ndarray,
    ) -> xp.ndarray:
        """Normalisation-specific wrapper when flats and darks are required.

        Args:
            method_name (str): The name of the method to use.
            params (Dict): A dict containing independent of httomo params.
            data (xp.ndarray): a numpy or cupy data array.
            flats (xp.ndarray): a numpy or cupy flats array.
            darks (xp.ndarray): a numpy or darks flats array.

        Returns:
            xp.ndarray: a numpy or cupy array of the normalised data.
        """
        # TODO: deal with ncore more accurately
        params.pop("ncore", None)  #: silently remove `ncore` param.

        if gpu_enabled:
            if self.cupyrun:
                # the method accepts CuPy arrays for the GPU processing
                with xp.cuda.Device(self.gpu_id):
                    # move the data to a particular device
                    data = xp.asarray(data)
                    flats = xp.asarray(flats)
                    darks = xp.asarray(darks)
            else:
                # the method doesn't accept CuPy arrays
                data = xp.asnumpy(data)
                flats = xp.asnumpy(flats)
                darks = xp.asnumpy(darks)

        data = getattr(self.module, method_name)(data, flats, darks, **params)
        return data

    def _execute_reconstruction(
        self,
        method_name: str,
        params: Dict,
        data: xp.ndarray,
        angles_radians: np.ndarray,
    ) -> xp.ndarray:
        """The reconstruction wrapper.

        Args:
            method_name (str): The name of the method to use.
            params (Dict): A dict containing independent of httomo params.
            data (xp.ndarray): a numpy or cupy data array.
            angles_radians (np.ndarray): a numpy array of projection angles.

        Returns:
            xp.ndarray: a numpy or cupy array of the reconstructed data.
        """
        # TODO: deal with ncore more accurately
        params.pop("ncore", None)  #: silently remove `ncore` param.

        if gpu_enabled:
            if self.cupyrun:
                # the method accepts CuPy arrays for the GPU processing
                with xp.cuda.Device(self.gpu_id):
                    # move the data to a particular device
                    data = xp.asarray(data)
            else:
                # the method doesn't accept CuPy arrays
                data = xp.asnumpy(data)

        # for 360 degrees data the angular dimension will be truncated while angles are not.
        # Truncating angles if the angular dimension has got a different size
        if data.shape[0] != len(angles_radians):
            angles_radians = angles_radians[0 : data.shape[0]]

        data = getattr(self.module, method_name)(data, angles_radians, **params)
        return data

    def _execute_rotation(
        self, method_name: str, params: Dict, data: xp.ndarray
    ) -> float:
        """The center of rotation wrapper.

        Args:
            method_name (str): The name of the method to use.
            params (Dict): A dict containing independent of httomo params.
            data (xp.ndarray): a numpy or cupy data array.

        Returns:
            float: The center of rotation.
        """
        # TODO: deal with ncore more accurately
        params.pop("ncore", None)  #: silently remove `ncore` param.

        if gpu_enabled:
            if self.cupyrun:
                # the method accepts CuPy arrays for the GPU processing
                with xp.cuda.Device(self.gpu_id):
                    # move the data to a particular device
                    data = xp.asarray(data)
            else:
                # the method doesn't accept CuPy arrays
                data = xp.asnumpy(data)

        method_func = getattr(self.module, method_name)
        rot_center = 0
        overlap = 0
        side = 0
        overlap_position = 0
        mid_rank = int(round(self.comm.size / 2) + 0.1)
        if self.comm.rank == mid_rank:
            if params["ind"] == "mid":
                params["ind"] = data.shape[1] // 2  # get the middle slice
            if method_name == "find_center_360":
                (rot_center, overlap, side, overlap_position) = method_func(
                    data, **params
                )
            else:
                rot_center = method_func(data, **params)

        rot_center = self.comm.bcast(rot_center, root=mid_rank)
        print_once(
            "The center of rotation for 180 degrees sinogram is {}".format(rot_center),
            self.comm,
            colour="cyan",
        )
        return rot_center


class TomoPyWrapper(BaseWrapper):
    """A class that wraps TomoPy functions for httomo"""

    def __init__(
        self, module_name: str, function_name: str, method_name: str, comm: Comm
    ):
        super().__init__(module_name, function_name, method_name, comm)

        # if not changed bellow the generic wrapper will be executed
        self.wrapper_method = super()._execute_generic

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
        self.wrapper_method = super()._execute_generic

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

        # httomolib can include GPU/CuPy methods as as well as CPU ones. Here
        # we check if the method can accept CuPy arrays.

        # get the docstring of a method in order to check the I/O requirements
        get_method_docs = inspect.getdoc(getattr(self.module, method_name))
        # if the CuPy array mentioned in the docstring then we will enable
        # the GPU run when it is possible
        self.cupyrun = False
        if "cp.ndarray" in get_method_docs:
            self.cupyrun = True

    def _execute_images(
        self, method_name: str, params: Dict, out_dir: str, comm: Comm, data: xp.ndarray
    ) -> None:
        """httomolib wrapper for save images function.

        Args:
            method_name (str): The name of the method to use.
            params (Dict): A dict containing independent of httomo params.
            out_dir (str): The output directory.
            comm (Comm): The MPI communicator.
            data (xp.ndarray): a numpy or cupy data array.

        Returns:
            None: returns None.
        """
        params.pop("ncore", None)  #: silently remove `ncore` param.

        if gpu_enabled:
            data = getattr(self.module, method_name)(
                xp.asnumpy(data), out_dir, comm_rank=comm.rank, **params
            )
        else:
            data = getattr(self.module, method_name)(
                data, out_dir, comm_rank=comm.rank, **params
            )
        return None
