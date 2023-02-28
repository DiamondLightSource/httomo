from typing import Dict
import inspect
import numpy as np
from mpi4py.MPI import Comm

gpu_enabled = False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
        gpu_enabled = True
    except cp.cuda.runtime.CUDARuntimeError:
        print("CuPy is installed but GPU device inaccessible")        
except ImportError:
    print("CuPy is not installed")   
  
class httomolib_wrapper:
    """_summary_
    """
    def __init__(self,
                 module_name: str,
                 function_name: str,
                 comm: Comm):
        # set the GPU id
        if gpu_enabled:
            self.num_GPUs = cp.cuda.runtime.getDeviceCount()
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

    def execute(self, method_name: str, params: Dict, data: np.ndarray):
        # get the docstring in order to check the CuPy dependency
        get_method_docs = inspect.getdoc(getattr(self.module, method_name))
        # if GPU is enabled and we need the method to be run on the GPU
        if gpu_enabled and "cp.ndarray" in get_method_docs:
            cp._default_memory_pool.free_all_blocks()
            cp.cuda.Device(self.gpu_id).use() # use a particular GPU by its id
            # TODO: do we need to convert a potential numy array to cupy here?
            with cp.cuda.Device(self.gpu_id):
                data = cp.asarray(data) # move the data to a particular device
                        
        # excute the method                    
        data = getattr(self.module, method_name)(data, **params)
        return data
        
class tomopy_wrapper:
    """_summary_
    """
    def __init__(self, module_name: str = None, function_name: str = None):
        if module_name == "misc":
            from tomopy import misc
            self.module = getattr(misc, function_name) # imports the module
        if module_name == "prep":
            from tomopy import prep
            self.module = getattr(prep, function_name) # imports the module            

    def execute(self, method_name: str, params: Dict, data: np.ndarray):
        # gets module executed
        data = getattr(self.module, method_name)(data, **params)
        return data