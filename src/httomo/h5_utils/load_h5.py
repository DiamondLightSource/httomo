import math

import h5py as h5
from mpi4py import MPI


def load_data(file, dim, path, preview=":,:,:", pad=(0, 0), comm=MPI.COMM_WORLD):
    """Load data in parallel, slicing it through a certain dimension.

    Args:
        file: Path to file containing the dataset.
        dim: Dimension along which data is sliced when being split between MPI
            processes.
        path: Path to dataset within the file.
        preview: Crop the data with a preview:
        pad: Pad the data by this number of slices.
        comm: MPI communicator object.
    """
    if dim == 1:
        data = read_through_dim1(file, path, preview=preview, pad=pad, comm=comm)
    elif dim == 2:
        data = read_through_dim2(file, path, preview=preview, pad=pad, comm=comm)
    elif dim == 3:
        data = read_through_dim3(file, path, preview=preview, pad=pad, comm=comm)
    else:
        raise Exception("Invalid dimension. Choose 1, 2 or 3.")
    return data


def read_through_dim3(file, path, preview=":,:,:", pad=(0, 0), comm=MPI.COMM_WORLD):
    """Read a dataset in parallel, with each MPI process loading a block.

    Args:
        file: Path to file containing the dataset.
        path: Path to dataset within the file.
        preview: Crop the data with a preview.
        pad: Pad the data by this number of slices.
        comm: MPI communicator object.
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        # Turning the preview into a length and offset. Data will be read from
        # data[offset] to data[offset + length].
        if slice_list[2] == slice(None):
            length = dataset.shape[2]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[2].start is None else slice_list[1].start
            stop = (
                dataset.shape[2] if slice_list[2].stop is None else slice_list[2].stop
            )
            step = 1 if slice_list[2].step is None else slice_list[2].step
            length = (
                stop - start
            ) // step  # Total length of the section of the dataset being read.
            offset = start  # Offset where the dataset will start being read.
        # Bounds of the data this process will load. Length is split between number of
        # processes.
        i0 = round((length / nproc) * rank) + offset - pad[0]
        i1 = round((length / nproc) * (rank + 1)) + offset + pad[1]
        # Checking that i0 and i1 are still within the bounds of the dataset after
        # padding.
        if i0 < 0:
            i0 = 0
        if i1 > dataset.shape[2]:
            i1 = dataset.shape[2]
        data = dataset[:, :, i0:i1:step]
        return data


def read_through_dim2(file, path, preview=":,:,:", pad=(0, 0), comm=MPI.COMM_WORLD):
    """Read a dataset in parallel, with each MPI process loading a block.

    Args:
        file: Path to file containing the dataset.
        path: Path to dataset within the file.
        preview: Crop the data with a preview.
        pad: Pad the data by this number of slices.
        comm: MPI communicator object.
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        # Turning the preview into a length and offset. Data will be read from
        # data[offset] to data[offset + length].
        if slice_list[1] == slice(None):
            length = dataset.shape[1]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[1].start is None else slice_list[1].start
            stop = (
                dataset.shape[1] if slice_list[1].stop is None else slice_list[1].stop
            )
            step = 1 if slice_list[1].step is None else slice_list[1].step
            length = (
                stop - start
            ) // step  # Total length of the section of the dataset being read.
            offset = start  # Offset where the dataset will start being read.
        # Bounds of the data this process will load. Length is split between number of
        # processes.
        i0 = round((length / nproc) * rank) + offset - pad[0]
        i1 = round((length / nproc) * (rank + 1)) + offset + pad[1]
        # Checking that i0 and i1 are still within the bounds of the dataset after
        # padding.
        if i0 < 0:
            i0 = 0
        if i1 > dataset.shape[1]:
            i1 = dataset.shape[1]
        data = dataset[slice_list[0], i0:i1:step, :]
        return data


def read_through_dim1(file, path, preview=":,:,:", pad=(0, 0), comm=MPI.COMM_WORLD):
    """Read a dataset in parallel, with each MPI process loading a block.

    Args:
        file: Path to file containing the dataset.
        path: Path to dataset within the file.
        preview: Crop the data with a preview.
        pad: Pad the data by this number of slices.
        comm: MPI communicator object.
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        # Turning the preview into a length and offset. Data will be read from
        # data[offset] to data[offset + length].
        if slice_list[0] == slice(None):
            length = dataset.shape[0]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[0].start is None else slice_list[0].start
            stop = (
                dataset.shape[0] if slice_list[0].stop is None else slice_list[0].stop
            )
            step = 1 if slice_list[0].step is None else slice_list[0].step
            length = (
                stop - start
            ) // step  # Total length of the section of the dataset being read.
            offset = start  # Offset where the dataset will start being read.
        # Bounds of the data this process will load. Length is split between number of
        # processes.
        i0 = round((length / nproc) * rank) + offset - pad[0]
        i1 = round((length / nproc) * (rank + 1)) + offset + pad[1]
        # Checking that i0 and i1 are still within the bounds of the dataset after
        # padding.
        if i0 < 0:
            i0 = 0
        if i1 > dataset.shape[0]:
            i1 = dataset.shape[0]
        data = dataset[i0:i1:step, slice_list[1], slice_list[2]]
        return data


def get_pad_values(
    pad, dim, dim_length, data_indices=None, preview=":,:,:", comm=MPI.COMM_WORLD
):
    """Get number of slices the block of data is padded either side.

    Args:
        pad: Number of slices to pad block with.
        dim: Dimension data is to be padded in (same dimension data is sliced in).
        dim_length: Size of dataset in the relevant dimension.
        data_indices: When a dataset has non-data in the dataset (for example darks &
            flats) provide data indices to indicate where in the dataset the data lies.
            Only has an effect when dim = 1, as darks and flats are in projection space.
        preview: Preview the data will be cropped by. Should be the same preview given
            to the get_data() method.
        comm: MPI communicator object.
    """
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    if data_indices is not None and dim == 1:
        bound0 = min(data_indices)
        bound1 = max(data_indices) + 1
    else:
        bound0 = 0
        bound1 = dim_length
    if slice_list[0] == slice(None):
        length = dim_length
        offset = 0
        step = 1
    else:
        start = 0 if slice_list[0].start is None else slice_list[0].start
        stop = dim_length if slice_list[0].stop is None else slice_list[0].stop
        step = 1 if slice_list[0].step is None else slice_list[0].step
        length = (
            stop - start
        ) // step  # Total length of the section of the dataset being read.
        offset = start  # Offset where the dataset will start being read.
    # i0, i1 = range of the data this process will load..
    i0 = round((length / nproc) * rank) + offset - pad
    i1 = round((length / nproc) * (rank + 1)) + offset + pad
    # Checking that after padding, the range is still within the bounds it should be.
    if i0 < bound0:
        pad0 = pad - (bound0 - i0)
    else:
        pad0 = pad
    if i1 > bound1:
        pad1 = pad - (i1 - bound1)
    else:
        pad1 = pad
    return pad0, pad1


def get_num_chunks(file, path, comm):
    """Gets the number of chunks in a file.

    Args:
        file: The hdf5 file to read from.
        path: The key of the dataset within the file.
        comm: The MPI communicator.
    """
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        shape = dataset.shape
        chunks = dataset.chunks

    chunk_boundaries = [
        [None] * (math.ceil(shape[i] / chunks[i]) + 1) for i in range(len(shape))
    ]

    # Creating a list of chunk boundaries in each dimension.
    for dim in range(len(shape)):
        boundary = 0
        for i in range(len(chunk_boundaries[dim])):
            if boundary > shape[dim]:
                boundary = shape[dim]
            chunk_boundaries[dim][i] = boundary
            boundary += chunks[dim]

    # Calculating number of chunks
    nchunks = 1
    for dim in range(len(chunk_boundaries)):
        nchunks *= len(chunk_boundaries[dim]) - 1

    return nchunks


def get_angles(
    file, path="/entry1/tomo_entry/data/rotation_angle", comm=MPI.COMM_WORLD
):
    """Get angles.

    Args:
        file: Path to file containing the data and angles.
        path: Path to the angles within the file.
        comm: MPI communicator object.
    """
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        angles = file[path][...]
    return angles


def get_darks_flats(
    file,
    data_path="/entry1/tomo_entry/data/data",
    image_key_path="/entry1/instrument/image_key/image_key",
    dim=1,
    pad=0,
    preview=":,:,:",
    comm=MPI.COMM_WORLD,
):
    """Get darks and flats.

    Args:
        file: Path to file containing the dataset.
        data_path: Path to the dataset within the file.
        image_key_path: Path to the image_key within the file.
        dim: Dimension along which data is being split between MPI processes. Only
            effects darks and flats if dim = 2.
        pad: How many slices data is being padded. Only effects darks and flats if
            dim = 2. (not implemented yet)
        preview: Preview data is being cropped by.
        comm: MPI communicator object.
    """
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        darks_indices = []
        flats_indices = []
        for i, key in enumerate(file[image_key_path]):
            if int(key) == 1:
                flats_indices.append(i)
            elif int(key) == 2:
                darks_indices.append(i)
        dataset = file[data_path]

        if dim == 2:
            rank = comm.rank
            nproc = comm.size
            if slice_list[1] == slice(None):
                length = dataset.shape[1]
                offset = 0
                step = 1
            else:
                start = 0 if slice_list[1].start is None else slice_list[1].start
                stop = (
                    dataset.shape[1]
                    if slice_list[1].stop is None
                    else slice_list[1].stop
                )
                step = 1 if slice_list[1].step is None else slice_list[1].step
                length = (stop - start) // step
                offset = start
            i0 = round((length / nproc) * rank) + offset
            i1 = round((length / nproc) * (rank + 1)) + offset
            darks = [dataset[x][i0:i1:step][slice_list[2]] for x in darks_indices]
            flats = [dataset[x][i0:i1:step][slice_list[2]] for x in flats_indices]
        else:
            darks = [
                file[data_path][x][slice_list[1]][slice_list[2]] for x in darks_indices
            ]
            flats = [
                file[data_path][x][slice_list[1]][slice_list[2]] for x in flats_indices
            ]
        return darks, flats


def get_data_indices(
    file, image_key_path="/entry1/instrument/image_key/image_key", comm=MPI.COMM_WORLD
):
    """Get the indices of where the data is in a dataset.

    Args:
        file: Path to the file containing the dataset and image key.
        image_key_path: Path to the image key within the file.
        comm: MPI communicator object.
    """
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        data_indices = []
        for i, key in enumerate(file[image_key_path]):
            if int(key) == 0:
                data_indices.append(i)
    return data_indices


def get_slice_list_from_preview(preview):
    """Generate slice list to crop data from a preview.

    Args:
        preview: Preview in the form 'start: stop: step, start: stop: step'.
    """
    slice_list = [None] * 3
    preview = preview.split(",")  # Splitting the dimensions
    for dimension, value in enumerate(preview):
        values = value.split(":")  # Splitting the start, stop, step
        new_values = [None if x.strip() == "" else int(x) for x in values]
        if len(values) == 1:
            slice_list[dimension] = slice(new_values[0])
        elif len(values) == 2:
            slice_list[dimension] = slice(new_values[0], new_values[1])
        elif len(values) == 3:
            slice_list[dimension] = slice(new_values[0], new_values[1], new_values[2])
    return slice_list
