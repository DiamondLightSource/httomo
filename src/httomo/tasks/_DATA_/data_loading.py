from pathlib import Path

from h5py import File
from mpi4py.MPI import Comm
from numpy import asarray, deg2rad, ndarray

from httomo.h5_utils import load_h5
from httomo.utils import print_once, print_rank


def load_data(
    in_file: Path, data_key: str, dimension: int, crop: int, pad: int, comm: Comm) -> tuple([ndarray, ndarray, ndarray, ndarray]):
    with File(in_file, "r", driver="mpio", comm=comm) as file:
        dataset = file[data_key]
        shape = dataset.shape
    print_once(f"Dataset shape is {shape}", comm)

    angles_degrees = load_h5.get_angles(in_file, comm=comm)
    data_indices = load_h5.get_data_indices(
        in_file,
        image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
        comm=comm,
    )
    angles = deg2rad(angles_degrees[data_indices])

    # preview to prepare to crop the data from the middle when --crop is used to
    # avoid loading the whole volume and crop out darks and flats when loading data.
    preview = [f"{data_indices[0]}: {data_indices[-1] + 1}", ":", ":"]
    if crop != 100:
        new_length = int(round(shape[1] * crop / 100))
        offset = int((shape[1] - new_length) / 2)
        preview[1] = f"{offset}: {offset + new_length}"
        cropped_shape = (
            data_indices[-1] + 1 - data_indices[0],
            new_length,
            shape[2],
        )
    else:
        cropped_shape = (data_indices[-1] + 1 - data_indices[0], shape[1], shape[2])
    preview = ", ".join(preview)

    print_once(f"Cropped data shape is {cropped_shape}", comm)

    dim = dimension
    pad_values = load_h5.get_pad_values(
        pad,
        dim,
        shape[dim - 1],
        data_indices=data_indices,
        preview=preview,
        comm=comm,
    )
    print_rank(f"Pad values are {pad_values}.", comm)
    data = load_h5.load_data(
        in_file, dim, data_key, preview=preview, pad=pad_values, comm=comm
    )

    darks, flats = load_h5.get_darks_flats(
        in_file,
        data_key,
        image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
        comm=comm,
        preview=preview,
        dim=dimension,
    )
    darks = asarray(darks)
    flats = asarray(flats)

    (angles_total, detector_y, detector_x) = data.shape
    print_rank(
        f"Data shape is {(angles_total, detector_y, detector_x)}"
        + f" of type {data.dtype}",
        comm,
    )

    return data, flats, darks, angles, angles_total, detector_y, detector_x
