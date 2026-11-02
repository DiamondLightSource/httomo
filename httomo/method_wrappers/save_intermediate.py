import os
from typing import Dict, Optional
import weakref
from mpi4py.MPI import Comm, MIN
import httomo
from httomo.block_interfaces import T
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.loader import LoaderInterface
from httomo.runner.method_wrapper import GpuTimeInfo, MethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import catchtime, xp, gpu_enabled

import h5py


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
        vds_per_rank: bool = True,
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
        self._vds_per_rank = vds_per_rank

        self._filename = f"{prev_method.task_id}-{prev_method.package_name}-{prev_method.method_name}"
        is_saving_recon = prev_method.module_path.endswith(".algorithm")
        if is_saving_recon and prev_method.recon_algorithm is not None:
            self._filename += f"-{prev_method.recon_algorithm}"
        if is_saving_recon and httomo.globals.RECON_FILENAME_STEM is not None:
            self._filename = httomo.globals.RECON_FILENAME_STEM
        self._global_filename = self._filename
        if self._vds_per_rank:
            self._filename += f"-rank-{comm.rank:02}"

        self._out_dir = out_dir
        if self._out_dir is None:
            self._out_dir = httomo.globals.run_out_dir
        assert self._out_dir is not None
        self._file = h5py.File(
            f"{self._out_dir}/{self._filename}.h5", "w", driver="mpio", comm=comm
        )
        # make sure file gets closed properly
        weakref.finalize(self, self._file.close)

    def _transfer_data(self, block: T) -> T:
        if block.is_cpu:
            return block
        if self._next_method_is_cpu:
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

        if self._next_method_is_cpu or not gpu_enabled:
            data = block.data_unpadded
        else:
            # Transfer data to CPU while the main block stays on GPU
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

        if self._vds_per_rank:
            global_shape = block.chunk_shape_unpadded
            global_index = block.chunk_index_unpadded
        else:
            global_shape = block.global_shape
            global_index = block.global_index_unpadded

        dataset_path = "/data"
        self._method(
            data,
            global_shape=global_shape,
            global_index=global_index,
            slicing_dim=block.slicing_dim,
            file=self._file,
            frames_per_chunk=httomo.globals.FRAMES_PER_CHUNK,
            minimum_block_length=self.config_params[MIN_BLOCK_LEN_PARAM],
            path=dataset_path,
            detector_x=self._loader.detector_x,
            detector_y=self._loader.detector_y,
            angles=block.angles,
        )

        if block.is_last_in_chunk:
            self._file.close()
            if self._vds_per_rank:
                global_file_path = f"{self._out_dir}/{self._global_filename}.h5"
                layout = h5py.VirtualLayout(
                    block.global_shape,
                    dtype=block.data.dtype,
                    filename=global_file_path,
                )
                virtual_source = h5py.VirtualSource(
                    path_or_dataset=f"{self._out_dir}/{self._filename}.h5",
                    name=dataset_path,
                    shape=block.chunk_shape_unpadded,
                )
                assert block.slicing_dim == 1
                layout[
                    :,
                    block.chunk_start : block.chunk_start + block.chunk_shape[1],
                    :,
                ] = virtual_source
                with h5py.File(
                    global_file_path, "w", driver="mpio", comm=self._comm
                ) as global_file:
                    global_file.create_virtual_dataset(name=dataset_path, layout=layout)

        return block
