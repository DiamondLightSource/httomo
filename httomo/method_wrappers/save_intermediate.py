import os
import pathlib
from typing import Any, Dict, Optional, Union
import weakref
from mpi4py.MPI import Comm, MIN
import httomo
from httomo.block_interfaces import T
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.loader import LoaderInterface
from httomo.runner.method_wrapper import GpuTimeInfo, MethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import catchtime, xp

import h5py
import numpy as np


class SaveIntermediateFilesWrapper(GenericMethodWrapper):
    @classmethod
    def should_select_this_class(cls, module_path: str, method_name: str) -> bool:
        return method_name == "save_intermediate_data"

    def __init__(
        self,
        method_repository: MethodRepository,
        module_path: str,
        method_name: str,
        comm: Comm,
        save_result: Optional[bool] = None,
        output_mapping: Dict[str, str] = {},
        out_dir: Optional[os.PathLike] = None,
        prev_method: Optional[MethodWrapper] = None,
        next_method_is_cpu: bool = False,
        loader: Optional[LoaderInterface] = None,
        **kwargs,
    ):
        super().__init__(
            method_repository,
            module_path,
            method_name,
            comm,
            save_result,
            output_mapping,
            **kwargs,
        )
        assert loader is not None
        self._loader = loader
        assert prev_method is not None
        self._next_method_is_cpu = next_method_is_cpu

        filename = f"{prev_method.task_id}-{prev_method.package_name}-{prev_method.method_name}"
        is_saving_recon = prev_method.module_path.endswith(".algorithm")
        if is_saving_recon and prev_method.recon_algorithm is not None:
            filename += f"-{prev_method.recon_algorithm}"
        if is_saving_recon and httomo.globals.RECON_FILENAME_STEM is not None:
            filename = httomo.globals.RECON_FILENAME_STEM

        if out_dir is None:
            out_dir = httomo.globals.run_out_dir
        assert out_dir is not None
        self._file = h5py.File(
            f"{out_dir}/{filename}.h5", "w", driver="mpio", comm=comm
        )
        # make sure file gets closed properly
        weakref.finalize(self, self._file.close)

    def _transfer_data(self, block: T) -> T:
        if block.is_cpu:
            return block
        if not self.cupyrun and self._next_method_is_cpu:
            # convert the whole (GPU) block to CPU if the next method is CPU
            self._gpu_time_info = GpuTimeInfo()
            with catchtime() as t:
                block.to_cpu()
            self._gpu_time_info.device2host = t.elapsed
            return block
        return block

    def execute(self, block: T) -> T:
        # we overwrite the most of the execute method here
        # we transfer the data to CPU only if the next method is CPU, otherwise we keep it on GPU
        # in case if save_intermediate is the last method we also keep the data on GPU
        block = self._transfer_data(block)
        if self._next_method_is_cpu:
            data = block.data_unpadded
        else:
            # we transfer the data to CPU while the main block stays on GPU
            self._gpu_time_info = GpuTimeInfo()
            with catchtime() as t:
                data = xp.asnumpy(block.data_unpadded)
            self._gpu_time_info.device2host += t.elapsed

        MIN_BLOCK_LEN_PARAM = "minimum_block_length"
        if block.chunk_index_unpadded[block.slicing_dim] == 0 and self.comm.size > 1:
            minimum_block_length = self.comm.reduce(
                self.config_params[MIN_BLOCK_LEN_PARAM], MIN
            )
            minimum_block_length = self.comm.bcast(minimum_block_length)
            self.append_config_params({MIN_BLOCK_LEN_PARAM: minimum_block_length})

        self._method(
            data,
            global_shape=block.global_shape,
            global_index=block.global_index_unpadded,
            slicing_dim=block.slicing_dim,
            file=self._file,
            frames_per_chunk=httomo.globals.FRAMES_PER_CHUNK,
            minimum_block_length=self.config_params[MIN_BLOCK_LEN_PARAM],
            path="/data",
            detector_x=self._loader.detector_x,
            detector_y=self._loader.detector_y,
            angles=block.angles,
        )

        if block.is_last_in_chunk:
            self._file.close()

        return block
