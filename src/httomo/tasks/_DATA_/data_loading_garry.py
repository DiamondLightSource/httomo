from enum import IntEnum, unique
from pathlib import Path

import hdf5plugin  # noqa: F401
from h5py import Dataset, File, Group, h5d, h5f, h5p
from mpi4py.MPI import Comm
from numpy import argwhere, array, empty, ndarray


def _open_file(path: Path, comm: Comm) -> File:
    file_access_plist = h5p.create(h5p.FILE_ACCESS)
    file_access_plist.set_fapl_mpio(comm, comm.info)
    file_access_plist.set_libver_bounds(h5f.LIBVER_LATEST, h5f.LIBVER_LATEST)

    file_id = h5f.open(str(path).encode("ascii"), h5f.ACC_RDONLY, file_access_plist)
    return File(file_id)


def _open_dataset(group: Group, name: str) -> Dataset:
    dataset_access_plist = h5p.create(h5p.DATASET_ACCESS)
    dataset_access_plist.set_chunk_cache(0, 0, 0)

    dataset_id = h5d.open(group.id, name.encode("ascii"), dataset_access_plist)
    return Dataset(dataset_id)


@unique
class ProjectionTypes(IntEnum):
    DATA = 0
    FLAT = 1
    DARK = 2


def load_data(path: Path, key: str) -> ndarray:
    with File(path) as file:
        return array(file[key])


def load_projections(
    path: Path,
    projection_type: ProjectionTypes,
    projection_key: str,
    projection_type_key: str,
) -> ndarray:
    with File(path) as file:
        projection_types = array(file[projection_type_key])
        projection_indices = argwhere(projection_types == projection_type.value)

        projection_dataset = file[projection_key]
        projection_shape = (
            len(projection_indices),
            *projection_dataset.shape[1:],
        )
        data = empty(projection_shape, projection_dataset.dtype)
        for slice_idx, projection_index in enumerate(projection_indices):
            source_sel = (
                slice(projection_index.item(), projection_index.item() + 1),
                *(slice(0, dim_size) for dim_size in projection_dataset.shape[1:]),
            )
            dest_sel = (
                slice(slice_idx, slice_idx + 1),
                *(slice(0, dim_size) for dim_size in projection_dataset.shape[1:]),
            )
            projection_dataset.read_direct(data, source_sel, dest_sel)

        return data


def load_projections_distributed(
    path: Path,
    projection_type: ProjectionTypes,
    projection_key: str,
    projection_type_key: str,
    comm: Comm,
) -> ndarray:
    group_name, dataset_name = projection_key.rsplit("/", 1)
    with _open_file(path, comm) as file:
        group = file[group_name]

        projection_types = array(file[projection_type_key])
        projection_indices = argwhere(projection_types == projection_type.value)
        worker_initial_frame_idx = int(len(projection_indices) / comm.size * comm.rank)
        worker_final_frame_idx = int(
            len(projection_indices) / comm.size * (comm.rank + 1)
        )
        worker_projection_indices = projection_indices[
            worker_initial_frame_idx:worker_final_frame_idx
        ]

        projection_dataset = _open_dataset(group, dataset_name)
        worker_projection_shape = (
            len(worker_projection_indices),
            *projection_dataset.shape[1:],
        )
        data = empty(worker_projection_shape, projection_dataset.dtype)
        for slice_idx, worker_projection_index in enumerate(worker_projection_indices):
            source_sel = (
                slice(
                    worker_projection_index.item(), worker_projection_index.item() + 1
                ),
                *(slice(0, dim_size) for dim_size in projection_dataset.shape[1:]),
            )
            dest_sel = (
                slice(slice_idx, slice_idx + 1),
                *(slice(0, dim_size) for dim_size in projection_dataset.shape[1:]),
            )
            projection_dataset.read_direct(data, source_sel, dest_sel)

        return data
