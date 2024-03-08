from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.dataset import DataSetBlock
from httomo.runner.method_wrapper import MethodParameterDictType
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import Pattern, catchtime, log_rank, xp


import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm


from typing import Any, Dict, Optional, Tuple, Union


class RotationWrapper(GenericMethodWrapper):
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

    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return module_path.endswith(".rotation")

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
        super().__init__(
            method_repository, module_path, method_name, comm, save_result, output_mapping, **kwargs
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
        self,
        dict_params: MethodParameterDictType,
        dataset: Optional[DataSetBlock] = None,
    ) -> Dict[str, Any]:
        assert (
            dataset is not None
        ), "Rotation wrapper cannot work without a dataset input"
        # if the function needs "ind" argument, and we either don't give it or have it as "mid",
        # we set it to the mid point of the data dim 1
        updated_params = dict_params
        if not "ind" in dict_params or (
            dict_params["ind"] == "mid" or dict_params["ind"] is None
        ):
            updated_params = {**dict_params, "ind": (dataset.shape[1] - 1) // 2}
        return super()._build_kwargs(updated_params, dataset)

    def _gather_sino_slice(self, global_shape: Tuple[int, int, int]):
        assert self.sino is not None
        if self.comm.size == 1:
            return self.sino

        # now aggregate with MPI
        sendbuf = self.sino.reshape(self.sino.size)
        sizes_rec = self.comm.gather(sendbuf.size)
        if self.comm.rank == 0:
            recvbuf = np.empty(global_shape[0] * global_shape[2], dtype=np.float32)
            self.comm.Gatherv(
                (sendbuf, self.sino.size, MPI.FLOAT),
                (recvbuf, sizes_rec, MPI.FLOAT),
                root=0,
            )
            return recvbuf.reshape((global_shape[0], global_shape[2]))
        else:
            self.comm.Gatherv(
                (sendbuf, self.sino.size, MPI.FLOAT),
                (None, sizes_rec, MPI.FLOAT),
                root=0,
            )
            return None

    def _run_method(self, block: DataSetBlock, args: Dict[str, Any]) -> DataSetBlock:
        assert "ind" in args
        slice_for_cor = args["ind"]
        # append to internal sinogram, until we have the last block
        if self.sino is None:
            self.sino = np.empty(
                (block.chunk_shape[0], block.chunk_shape[2]), dtype=np.float32
            )
        data = block.data[:, slice_for_cor, :]
        if block.is_gpu:
            with catchtime() as t:
                data = xp.asnumpy(data)
            self._gpu_time_info.device2host += t.elapsed
        self.sino[
            block.chunk_index[0] : block.chunk_index[0] + block.shape[0], :
        ] = data

        if not block.is_last_in_chunk:  # exit if we didn't process all blocks yet
            return block

        sino_slice = self._gather_sino_slice(block.chunk_shape)

        # now calculate the center of rotation on rank 0 and broadcast
        res: Optional[Union[tuple, float, np.float32]] = None
        if self.comm.rank == 0:
            sino_slice = self.normalize_sino(
                sino_slice,
                block.flats[:, slice_for_cor, :],
                block.darks[:, slice_for_cor, :],
            )
            if self.cupyrun:
                with catchtime() as t:
                    sino_slice = xp.asarray(sino_slice)
                self._gpu_time_info.host2device += t.elapsed
            args["ind"] = 0
            args[self.parameters[0]] = sino_slice
            res = self.method(**args)
        if self.comm.size > 1:
            res = self.comm.bcast(res, root=0)

        cor_str = f"The center of rotation for sinogram is {res}"
        log_rank(cor_str, comm=self.comm)
        return self._process_return_type(res, block)

    def normalize_sino(
        self, sino: np.ndarray, flats: Optional[np.ndarray], darks: Optional[np.ndarray]
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
            with catchtime() as t:
                sino = xp.asarray(sino)
            self._gpu_time_info.host2device += t.elapsed
            sino -= darks1d / denom
        else:
            # CPU
            denom[np.where(denom == 0.0)] = 1.0
            sino -= darks1d / denom
        return sino[:, np.newaxis, :]

    def _process_return_type(
        self, ret: Any, input_block: DataSetBlock
    ) -> DataSetBlock:
        if type(ret) == tuple:
            # cor, overlap, side, overlap_position - from find_center_360
            self._side_output["cor"] = float(ret[0])
            self._side_output["overlap"] = float(ret[1])
            self._side_output["side"] = int(ret[2]) if ret[2] is not None else None
            self._side_output["overlap_position"] = float(ret[3])  # float
        else:
            self._side_output["cor"] = float(ret)
        return input_block
