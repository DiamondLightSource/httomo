from mpi4py import MPI
from httomo._stats.globals import min_max_mean_std
import httomo.globals
from httomo.data import mpiutil
from httomo.runner.dataset import DataSet, DataSetBlock
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.runner.output_ref import OutputRef
from httomo.utils import Colour, Pattern, gpu_enabled, log_once, log_rank, xp
from httomo.runner.gpu_utils import gpumem_cleanup
from httomo.data.hdf._utils.reslice import single_sino_reslice

import numpy as np
from mpi4py.MPI import Comm

import logging
import os
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)


class BackendWrapper:
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
        output_mapping: Dict[str, str] = {},
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
        output_mapping: Dict[str, str]
            Mapping of side-output names to new ones. Used for propagating side outputs by name.
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
        self.parameters = [
            k for k, p in sig.parameters.items() if p.kind != Parameter.VAR_KEYWORD
        ]
        self._params_with_defaults = [
            k for k, p in sig.parameters.items() if p.default != Parameter.empty
        ]
        self._has_kwargs = any(
            p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        # check if the given kwargs are actually supported by the method
        self._config_params = kwargs
        self._output_mapping = output_mapping
        self._check_config_params()

        # assign method properties from the methods repository
        self.query = method_repository.query(module_path, method_name)
        self.pattern = self.query.get_pattern()
        self.output_dims_change = self.query.get_output_dims_change()
        self.implementation = self.query.get_implementation()
        self.memory_gpu = self.query.get_memory_gpu_params()

        if self.is_gpu and not gpu_enabled:
            raise ValueError("GPU is not available, please use only CPU methods")

        self._side_output: Dict[str, Any] = dict()

        if gpu_enabled:
            self.num_gpus = xp.cuda.runtime.getDeviceCount()
            _id = httomo.globals.gpu_id
            self.gpu_id = mpiutil.local_rank % self.num_gpus if _id == -1 else _id

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
    def package_name(self) -> str:
        return self.module_path.split(".")[0]

    def __getitem__(self, key: str) -> DictValues:
        """Get a parameter for the method using dictionary notation (wrapper["param"])"""
        return self._config_params[key]

    def __setitem__(self, key: str, value: DictValues):
        """Set a parameter for the method using dictionary notation (wrapper["param"] = 42)"""
        self._config_params[key] = value
        self._check_config_params()

    @property
    def config_params(self) -> Dict[str, Any]:
        """Access a copy of the configuration parameters (cannot be modified directly)"""
        return {**self._config_params}

    def append_config_params(self, params: DictType):
        """Append configuration parameters to the existing config_params"""
        self._config_params |= params
        self._check_config_params()

    def _check_config_params(self):
        if self._has_kwargs:
            return
        for k in self._config_params.keys():
            if k not in self.parameters:
                raise ValueError(
                    f"{self.method_name}: Unsupported keyword argument given: {k}"
                )

    @property
    def recon_algorithm(self) -> Optional[str]:
        """Determine the recon algorithm used, if the method is reconstruction.
        Otherwise return None."""
        return None

    def _build_kwargs(
        self, dict_params: DictType, dataset: Optional[DataSet] = None
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
                ret[p] = self.gpu_id
            elif p == "glob_stats":
                if remaining_dict_params.get(p, False):
                    assert dataset is not None
                    dataset.to_cpu()
                    ret[p] = min_max_mean_std(dataset.data, self.comm)
                else:
                    assert p in self._params_with_defaults
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
        if self.query.swap_dims_on_output():
            ret = ret.swapaxes(0, 1)
        input_dataset.data = ret
        return input_dataset

    def get_side_output(self) -> Dict[str, Any]:
        """Override this method for functions that have a side output. The returned dictionary
        will be merged with the dict_params parameter passed to execute for all methods that
        follow in the pipeline"""
        return {v: self._side_output[k] for k, v in self._output_mapping.items()}

    def _transfer_data(self, dataset: DataSet):
        if not self.cupyrun:
            dataset.to_cpu()
            return dataset

        assert gpu_enabled, "GPU method used on a system without GPU support"

        xp.cuda.Device(self.gpu_id).use()
        gpulog_str = f"Using GPU {self.gpu_id} to transfer data of shape {xp.shape(dataset.data[0])}"
        log_rank(gpulog_str, comm=self.comm)
        gpumem_cleanup()
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

    def calculate_output_dims(
        self, non_slice_dims_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate the dimensions of the output for this method"""
        if self.output_dims_change:
            return self.query.calculate_output_dims(
                non_slice_dims_shape, **self.config_params
            )

        return non_slice_dims_shape

    def calculate_max_slices(
        self,
        dataset: DataSet,
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
        if len(self.memory_gpu) == 0:
            return (
                available_memory
                // (np.prod(non_slice_dims_shape) * dataset.data.itemsize),
                available_memory,
            )

        # NOTE: This could go directly into the methodquery / method database,
        # and here we just call calculated_memory_bytes
        memory_bytes_method = 0
        for field in self.memory_gpu:
            subtract_bytes = 0
            # loop over the dataset names given in the library file and extracting
            # the corresponding dimensions from the available datasets
            if field.dataset in ["flats", "darks"]:
                # for normalisation module dealing with flats and darks separately
                array: np.ndarray = getattr(dataset, field.dataset)
                available_memory -= int(field.multiplier * array.nbytes)
            else:
                # deal with the rest of the data
                if field.method == "direct":
                    # this calculation assumes a direct (simple) correspondence through multiplier
                    memory_bytes_method += int(
                        field.multiplier
                        * np.prod(non_slice_dims_shape)
                        * dataset.data.itemsize
                    )
                else:
                    (
                        memory_bytes_method,
                        subtract_bytes,
                    ) = self.query.calculate_memory_bytes(
                        non_slice_dims_shape, dataset.data.dtype, **self.config_params
                    )

        if memory_bytes_method == 0:
            return available_memory - subtract_bytes, available_memory
        
        return (
            available_memory - subtract_bytes
        ) // memory_bytes_method, available_memory


class ReconstructionWrapper(BackendWrapper):
    """Wraps reconstruction functions, limiting the length of the angles array
    before calling the method."""

    def _preprocess_data(self, dataset: DataSet) -> DataSet:
        # for 360 degrees data the angular dimension will be truncated while angles are not.
        # Truncating angles if the angular dimension has got a different size
        datashape0 = dataset.data.shape[0]
        if datashape0 != len(dataset.angles_radians):
            dataset.unlock()
            dataset.angles_radians = dataset.angles_radians[0:datashape0]
            dataset.lock()
        self._input_shape = dataset.data.shape
        return super()._preprocess_data(dataset)

    def _build_kwargs(
        self, dict_params: BackendWrapper.DictType, dataset: DataSet | None = None
    ) -> Dict[str, Any]:
        # for recon methods, we assume that the second parameter is the angles in all cases
        assert (
            len(self.parameters) >= 2
        ), "recon methods always take data + angles as the first 2 parameters"
        updated_params = {**dict_params, self.parameters[1]: dataset.angles_radians}
        return super()._build_kwargs(updated_params, dataset)

    @property
    def recon_algorithm(self) -> Optional[str]:
        assert "center" in self.parameters, (
            "All recon methods should have a 'center' parameter, but it doesn't seem"
            + f" to be the case for {self.module_path}.{self.method_name}"
        )
        return self._config_params.get("algorithm", None)


class RotationWrapper(BackendWrapper):
    """Wraps rotation (centering) methods, which output the original dataset untouched,
    but have a side output for the center of rotation data (handling both 180 and 360).

    It wraps the actual algorithm to find the center and does more. In particular:
    - takes a single sinogram from the full dataset (across all MPI processes)
    - normalises it
    - calls the center-finding algorithm on this normalised data slice
    - outputs the center of rotation as a side output

    For block-wise processing support, it accumulates the sinogram in-memory in the method
    until the sinogram is complete for the current process. Then it uses MPI to add the data
    from the other processes to it.
    """

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        output_mapping: Dict[str, str] = {},
        **kwargs,
    ):
        super().__init__(
            method_repository, module_path, method_name, comm, output_mapping, **kwargs
        )
        if self.pattern not in [Pattern.sinogram, Pattern.all]:
            raise NotImplementedError(
                "Base method for rotation wrapper only supports sinogram or all"
            )
        # we must use projection in this wrapper now, to determine the full sino slice with MPI
        # TODO: extend this to support Pattern.all
        self.pattern = Pattern.projection
        self.sino: Optional[np.ndarray] = None

    def _build_kwargs(
        self, dict_params: BackendWrapper.DictType, dataset: DataSet | None = None
    ) -> Dict[str, Any]:
        # if the function needs "ind" argument, and we either don't give it or have it as "mid",
        # we set it to the mid point of the data dim 1
        updated_params = dict_params
        if not "ind" in dict_params or (
            dict_params["ind"] == "mid" or dict_params["ind"] is None
        ):
            updated_params = {**dict_params, "ind": (dataset.global_shape[1] - 1) // 2}
        return super()._build_kwargs(updated_params, dataset)

    def _gather_sino_slice(self, global_shape: Tuple[int, int]):
        if self.comm.size == 1:
            return self.sino

        # now aggregate with MPI
        if self.comm.rank == 0:
            recvbuf = np.empty(global_shape[0] * global_shape[2], dtype=np.float32)
        else:
            recvbuf = None
        sendbuf = self.sino.reshape(self.sino.size)
        sizes_rec = self.comm.gather(sendbuf.size)
        self.comm.Gatherv(
            (sendbuf, self.sino.size, MPI.FLOAT),
            (recvbuf, sizes_rec, MPI.FLOAT),
            root=0,
        )
        if self.comm.rank == 0:
            return recvbuf.reshape((global_shape[0], global_shape[2]))
        else:
            return None

    def _run_method(self, dataset: DataSet, args: Dict[str, Any]) -> DataSet:
        assert "ind" in args
        slice_for_cor = args["ind"]
        # append to internal sinogram, until we have the last block
        if self.sino is None:
            self.sino = np.empty(
                (dataset.chunk_shape[0], dataset.chunk_shape[2]), dtype=np.float32
            )
        data = dataset.data[:, slice_for_cor, :]
        if dataset.is_gpu:
            data = data.get()
        self.sino[
            dataset.chunk_index[0] : dataset.chunk_index[0] + dataset.shape[0], :
        ] = data

        if not dataset.is_last_in_chunk:  # exit if we didn't process all blocks yet
            return dataset

        sino_slice = self._gather_sino_slice(dataset.global_shape)

        # now calculate the center of rotation on rank 0 and broadcast
        res: Optional[Union[tuple, float, np.float32]] = None
        if self.comm.rank == 0:
            sino_slice = self.normalize_sino(
                sino_slice,
                dataset.flats[:, slice_for_cor, :],
                dataset.darks[:, slice_for_cor, :],
            )
            if self.cupyrun:
                sino_slice = xp.asarray(sino_slice)
            args["ind"] = 0
            args[self.parameters[0]] = sino_slice
            res = self.method(**args)
        if self.comm.size > 1:
            res = self.comm.bcast(res, root=0)
        return self._process_return_type(res, dataset)

    @classmethod
    def normalize_sino(
        cls, sino: np.ndarray, flats: Optional[np.ndarray], darks: Optional[np.ndarray]
    ) -> np.ndarray:
        flats1d = 1.0 if flats is None else flats.mean(0, dtype=np.float32)
        darks1d = 0.0 if darks is None else darks.mean(0, dtype=np.float32)
        denom = flats1d - darks1d
        sino = sino.astype(np.float32)
        if np.shape(denom) == tuple():
            sino -= darks1d  # denominator is always 1
        elif getattr(denom, "device", None) is not None:
            # GPU
            denom[xp.where(denom == 0.0)] = 1.0
            sino = xp.asarray(sino)
            sino -= darks1d / denom
        else:
            # CPU
            denom[np.where(denom == 0.0)] = 1.0
            sino -= darks1d / denom
        return sino[:, np.newaxis, :]

    def _process_return_type(self, ret: Any, input_dataset: DataSet) -> DataSet:
        if type(ret) == tuple:
            # cor, overlap, side, overlap_position - from find_center_360
            self._side_output["cor"] = float(ret[0])
            self._side_output["overlap"] = float(ret[1])
            self._side_output["side"] = int(ret[2]) if ret[2] is not None else None
            self._side_output["overlap_position"] = float(ret[3])  # float
        else:
            self._side_output["cor"] = float(ret)
        return input_dataset


class DezingingWrapper(BackendWrapper):
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
        output_mapping: Dict[str, str] = {},
        **kwargs,
    ):
        super().__init__(
            method_repository, module_path, method_name, comm, output_mapping, **kwargs
        )
        assert (
            method_name == "remove_outlier3d"
        ), "Only remove_outlier3d is supported at the moment"
        self._flats_darks_processed = False

    def execute(self, dataset: Union[DataSetBlock, DataSet]) -> DataSet:
        # check if data needs to be transfered host <-> device
        dataset = self._transfer_data(dataset)

        dataset.data = self.method(dataset.data, **self._config_params)
        if not self._flats_darks_processed:
            darks = self.method(dataset.darks, **self._config_params)
            flats = self.method(dataset.flats, **self._config_params)
            ds = dataset.base if isinstance(dataset, DataSetBlock) else dataset
            ds.unlock()
            ds.darks = darks
            ds.flats = flats
            ds.lock()
            self._flats_darks_processed = True

        return dataset


class ImagesWrapper(BackendWrapper):
    """Wraps image writer methods, which accept numpy (CPU) arrays as input,
    but don't actually modify the dataset. They write the information to files"""

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        out_dir: os.PathLike,
        output_mapping: Dict[str, str] = {},
        **kwargs,
    ):
        super().__init__(
            method_repository, module_path, method_name, comm, output_mapping, **kwargs
        )
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
    output_mapping: Dict[str, str] = {},
    **kwargs,
) -> BackendWrapper:
    """Factory function to generate the appropriate wrapper based on the module
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
    output_mapping: Dict[str, str]
        A dictionary mapping output names to translated ones. The side outputs will be renamed
        as specified, if the parameter is given. If not, no side outputs will be passed on.
    kwargs:
        Arbitrary keyword arguments that get passed to the method as parameters.

    Returns
    -------

    BackendWrapper
        An instance of a wrapper class
    """

    if module_path.endswith(".algorithm"):
        return ReconstructionWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            output_mapping=output_mapping,
            **kwargs,
        )
    if module_path.endswith(".rotation"):
        return RotationWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            output_mapping=output_mapping,
            **kwargs,
        )
    if method_name == "remove_outlier3d":
        return DezingingWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            output_mapping=output_mapping,
            **kwargs,
        )
    if module_path.endswith(".images"):
        return ImagesWrapper(
            method_repository=method_repository,
            module_path=module_path,
            method_name=method_name,
            comm=comm,
            output_mapping=output_mapping,
            out_dir=httomo.globals.run_out_dir,
            **kwargs,
        )
    return BackendWrapper(
        method_repository=method_repository,
        module_path=module_path,
        method_name=method_name,
        comm=comm,
        output_mapping=output_mapping,
        **kwargs,
    )
