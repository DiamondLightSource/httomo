import os
import pathlib
from typing import Dict, Optional, Literal
import weakref
from mpi4py.MPI import Comm, MIN
import httomo
from httomo.block_interfaces import T
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.runner.loader import LoaderInterface
from httomo.runner.method_wrapper import GpuTimeInfo, MethodWrapper
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import catchtime, xp, gpu_enabled, log_rank
import numpy as np
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
        use_virtual_dataset: Literal["yes", "no", "auto"] = "auto",
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

        if use_virtual_dataset == "yes":
            vds_per_rank = True
        elif use_virtual_dataset == "auto" and "SLURM_JOB_NUM_NODES" in os.environ:
            vds_per_rank = int(os.environ["SLURM_JOB_NUM_NODES"]) > 1
        else:
            vds_per_rank = False

        self._data_file = self._file
        if vds_per_rank:
            self._data_file = h5py.File(
                f"{out_dir}/{filename}-rank{comm.rank:02}.h5", "w"
            )
            log_rank(
                f'Writing output data to separate file at "{self._data_file.filename}"',
                comm=comm,
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
        self._save_auxiliary_data_hdf5(
            block.angles, self._loader.detector_x, self._loader.detector_y
        )
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

        vds_per_rank = self._data_file != self._file
        if vds_per_rank:
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
            file=self._data_file,
            frames_per_chunk=httomo.globals.FRAMES_PER_CHUNK,
            minimum_block_length=self.config_params[MIN_BLOCK_LEN_PARAM],
            path=dataset_path,
        )

        if block.is_last_in_chunk:
            if vds_per_rank:
                data_file_path = self._data_file.filename
                self._data_file.close()
                self._build_virtual_dataset(block, data_file_path, dataset_path)
            self._file.close()
        return block

    def _save_auxiliary_data_hdf5(
        self,
        angles: np.ndarray,
        detector_x: int,
        detector_y: int,
    ):
        if "/angles" in self._file:
            return

        self._file.create_dataset("angles", data=angles)

        file_name = pathlib.Path(self._file.filename).name
        self._file.create_dataset(file_name, data=[0, 0])
        g1 = self._file.create_group("data_dims")
        g1.create_dataset("detector_x_y", data=[detector_x, detector_y])

    def _build_virtual_dataset(self, block, data_file_path: str, dataset_path: str):
        layout = h5py.VirtualLayout(
            block.global_shape,
            dtype=block.data.dtype,
            filename=self._file.filename,
        )
        chunk_starts = self.comm.allgather(block.chunk_start)
        chunk_shapes = self.comm.allgather(block.chunk_shape_unpadded)
        data_file_paths = self.comm.allgather(data_file_path)

        for i in range(self.comm.size):
            virtual_source = h5py.VirtualSource(
                path_or_dataset=data_file_paths[i],
                name=dataset_path,
                shape=chunk_shapes[i],
            )

            index = [slice(None)] * len(layout.shape)
            index[block.slicing_dim] = slice(
                chunk_starts[i],
                chunk_starts[i] + chunk_shapes[i][block.slicing_dim],
            )
            layout[tuple(index)] = virtual_source
        self._file.create_virtual_dataset(name=dataset_path, layout=layout)
