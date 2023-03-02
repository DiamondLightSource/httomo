from typing import Dict
import numpy as np
import inspect
from mpi4py.MPI import Comm
from inspect import signature

gpu_enabled = False
try:
    import cupy as xp
    try:
        xp.cuda.Device(0).compute_capability
        gpu_enabled = True
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp
        print("CuPy is installed but GPU device inaccessible")        
except ImportError:
    import numpy as xp
    print("CuPy is not installed")
    
class wrappers:
    """A parent class for all wrappers in httomo that use external modules.
    """
    def __init__(self,
                 module_name: str = None,
                 function_name: str = None,
                 method_name: str = None):
        self.function_name = function_name
        
    def _execute_generic(self,
                         method_name: str,
                         params: Dict,
                         data: xp.ndarray) -> xp.ndarray:
        """The generic wrapper to execute functions for external packages.

        Args:
            method_name (str): The name of the method to use
            params (Dict): A dict containing independent of httomo params        
            data (xp.ndarray): a numpy or cupy data array

        Returns:
            xp.ndarray: A numpy or cupy array containing processed data.           
        """
        # gets module executed
        data = getattr(self.module, method_name)(data, **params)
        return data
    
    def _execute_normalize(self,
                          method_name: str,
                          params: Dict,
                          data: xp.ndarray,
                          flats: xp.ndarray,
                          darks: xp.ndarray) -> xp.ndarray:
        """For normalisation function we also require flats and darks.

        Args:
            method_name (str): The name of the method to use
            params (Dict): A dict containing independent of httomo params
            data (xp.ndarray): a numpy or cupy data array
            flats (xp.ndarray): a numpy or cupy flats array
            darks (xp.ndarray): a numpy or darks flats array

        Returns:
            xp.ndarray: a numpy or cupy array of normalised data
        """
        data = getattr(self.module, method_name)(data, flats, darks, **params)
        return data

    def _execute_reconstruction(self, 
                               method_name: str,
                               params: Dict,
                               data: xp.ndarray,
                               angles_radians: np.ndarray) -> xp.ndarray:
        """The reconstruction wrapper.

        Args:
            method_name (str): The name of the method to use
            params (Dict): A dict containing independent of httomo params
            data (xp.ndarray): a numpy or cupy data array
            angles_radians (np.ndarray): a numpy array of projection angles

        Returns:
            xp.ndarray: a numpy or cupy array of the reconstructed data
        """
        # for 360 degrees data the angular dimension will be truncated while angles are not.
        # Truncating angles if the angular dimension has got a different size
        angular_dim_size = np.size(data, 0)
        if np.size(data, 0) != len(angles_radians):
            angles_radians = angles_radians[0:angular_dim_size]
                
        data = getattr(self.module, method_name)(data, angles_radians, **params)
        return data   
    
class tomopy_wrapper(wrappers):
    """A class that wraps TomoPy functions for httomo
    """
    def __init__(self,
                 module_name: str = None,
                 function_name: str = None,
                 method_name: str = None):      
        super().__init__(module_name, function_name, method_name)
        
        self.wrapper_method = super(tomopy_wrapper, self)._execute_generic
        self.function_name = function_name
        self.method_name = method_name
        if module_name == "misc":
            from tomopy import misc
            self.module = getattr(misc, function_name)
        if module_name == "prep":
            from tomopy import prep
            self.module = getattr(prep, function_name)
            if function_name == "normalize":
                func = getattr(self.module, method_name)
                sig_params = signature(func).parameters
                if 'dark' in sig_params and 'flat' in sig_params:
                    self.wrapper_method = super(tomopy_wrapper, self)._execute_normalize
        if module_name == "recon":
            from tomopy import recon
            # a workaround to get the correct module, TomoPy incosistency?
            from importlib import import_module
            recon = import_module('tomopy.recon')
            self.module = getattr(recon, function_name)
            if function_name == "algorithm":
                self.wrapper_method = super(tomopy_wrapper, self)._execute_reconstruction

class httomolib_wrapper:
    """A class that wraps httomolib functions for httomo
    """
    def __init__(self,
                 module_name: str,
                 function_name: str,
                 comm: Comm):
        # set the GPU id
        self.comm = comm
        if gpu_enabled:
            self.num_GPUs = xp.cuda.runtime.getDeviceCount()
            self.gpu_id = int(comm.rank / comm.size * self.num_GPUs)
        if module_name == "misc":
            from httomolib import misc
            self.module = getattr(misc, function_name) # imports the module
        elif module_name == "prep":
            from httomolib import prep
            self.module = getattr(prep, function_name) # imports the module
        elif module_name == "recon":
            from httomolib import recon
            self.module = getattr(recon, function_name) # imports the module
        else:
            err_str = f"An unknown module name was encountered: " \
                      f"{module_name}"
            raise ValueError(err_str)

# TODO: execute method accepts only numpy arrays but it should be CPU/GPU agnostic
# OR there should be two separate methods execute_numpy and execute_cupy
# Any advice? 

    def execute(self,
                method_name: str,
                params: Dict,
                data: xp.ndarray):
        # get the docstring in order to check the CuPy dependency
        get_method_docs = inspect.getdoc(getattr(self.module, method_name))
        # if GPU is enabled and we need the method to be run on the GPU
        if gpu_enabled and "cp.ndarray" in get_method_docs:
            xp._default_memory_pool.free_all_blocks()
            xp.cuda.Device(self.gpu_id).use() # use a particular GPU by its id
            with xp.cuda.Device(self.gpu_id):
                data = xp.asarray(data) # move the data to a particular device
                        
        # excute the method
        data = getattr(self.module, method_name)(data, **params)
        return data