from os import PathLike
from typing import List
from unittest.mock import ANY
import numpy as np
import pytest
from pytest_mock import MockerFixture
from httomo.data.dataset_store import DataSetStoreReader, DataSetStoreWriter
from mpi4py import MPI

from httomo.runner.dataset import DataSet, FullFileDataSet


def test_writer_can_set_sizes_and_shapes_dim0(tmp_path: PathLike):
    writer = DataSetStoreWriter(
        full_size=30,
        slicing_dim=0,
        other_dims=(10, 20),
        chunk_size=10,
        chunk_start=10,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )

    assert writer.global_shape == (30, 10, 20)
    assert writer.chunk_shape == (10, 10, 20)
    assert writer.chunk_index == (10, 0, 0)
    assert writer.slicing_dim == 0


def test_writer_can_set_sizes_and_shapes_dim1(tmp_path: PathLike):
    writer = DataSetStoreWriter(
        full_size=30,
        slicing_dim=1,
        other_dims=(10, 20),
        chunk_size=10,
        chunk_start=10,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )

    assert writer.global_shape == (10, 30, 20)
    assert writer.chunk_shape == (10, 10, 20)
    assert writer.chunk_index == (0, 10, 0)
    assert writer.slicing_dim == 1


def test_reader_throws_if_no_data(dummy_dataset: DataSet, tmp_path: PathLike):
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    with pytest.raises(ValueError) as e:
        writer.make_reader()

    assert "no data" in str(e)


@pytest.mark.parametrize("file_based", [False, True])
def test_can_write_and_read_blocks(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike, file_based: bool
):
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block1 = dummy_dataset.make_block(0, 0, 2)
    block2 = dummy_dataset.make_block(0, 2, 2)

    if file_based:
        mocker.patch.object(writer, "_create_numpy_data", side_effect=MemoryError)
    writer.write_block(block1)
    writer.write_block(block2)

    reader = writer.make_reader()

    rblock1 = reader.read_block(0, 2)
    rblock2 = reader.read_block(2, 2)

    assert reader.global_shape == dummy_dataset.shape
    assert reader.chunk_shape == (4, dummy_dataset.shape[1], dummy_dataset.shape[2])
    assert reader.chunk_index == (3, 0, 0)
    assert reader.slicing_dim == 0

    assert isinstance(rblock1.data, np.ndarray)
    assert isinstance(rblock2.data, np.ndarray)

    np.testing.assert_array_equal(rblock1.data, block1.data)
    np.testing.assert_array_equal(rblock2.data, block2.data)


@pytest.mark.parametrize("file_based", [False, True])
def test_write_after_read_throws(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike, file_based: bool
):
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block = dummy_dataset.make_block(0, 0, 4)

    if file_based:
        mocker.patch.object(writer, "_create_numpy_data", side_effect=MemoryError)
    writer.write_block(block)

    writer.make_reader()

    with pytest.raises(ValueError):
        writer.write_block(block)


def test_writer_closes_file_on_finalize(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike
):
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block = dummy_dataset.make_block(0, 0, 4)

    mocker.patch.object(writer, "_create_numpy_data", side_effect=MemoryError)
    writer.write_block(block)
    fileclose = mocker.patch.object(writer._h5file, "close")
    writer.finalize()

    fileclose.assert_called_once()


def test_making_reader_closes_file_and_deletes(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike
):
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block = dummy_dataset.make_block(0, 0, 4)
    mocker.patch.object(writer, "_create_numpy_data", side_effect=MemoryError)
    writer.write_block(block)

    reader = writer.make_reader()

    assert writer._h5file is None
    assert writer.filename is not None
    assert writer.filename.exists()
    assert reader.filename == writer.filename
    assert reader._h5file is not None
    assert reader._h5file.get("data", None) is not None

    reader.finalize()

    assert not writer.filename.exists()


def test_can_write_and_read_block_with_different_sizes(
    dummy_dataset: DataSet, tmp_path: PathLike
):
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block1 = dummy_dataset.make_block(0, 0, 2)
    block2 = dummy_dataset.make_block(0, 2, 2)

    writer.write_block(block1)
    writer.write_block(block2)

    reader = writer.make_reader()

    rblock = reader.read_block(0, 4)

    np.testing.assert_array_equal(rblock.data, dummy_dataset.data[0:4, :, :])


def test_writing_too_large_blocks_fails(dummy_dataset: DataSet, tmp_path: PathLike):
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block = dummy_dataset.make_block(0, 2, 3)

    with pytest.raises(ValueError) as e:
        writer.write_block(block)

    assert "outside the chunk dimension" in str(e)


def test_writing_inconsistent_block_shapes_fails(
    dummy_dataset: DataSet, tmp_path: PathLike
):
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block1 = dummy_dataset.make_block(0, 0, 2)
    block2 = dummy_dataset.make_block(0, 2, 2)
    b2shape = list(block2.data.shape)
    b2shape[2] += 3
    block2.data = np.ones(b2shape, dtype=np.float32)

    writer.write_block(block1)
    with pytest.raises(ValueError) as e:
        writer.write_block(block2)

    assert "inconsistent shape" in str(e)


def test_create_new_data_goes_to_file_on_memory_error(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike
):
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )

    block1 = dummy_dataset.make_block(0, 0, 2)

    mocker.patch.object(writer, "_create_numpy_data", side_effect=MemoryError)
    createh5_mock = mocker.patch.object(
        writer, "_create_h5_data", return_value=dummy_dataset.data
    )

    writer.write_block(block1)

    createh5_mock.assert_called_with(
        writer.global_shape,
        dummy_dataset.data.dtype,
        ANY,
        writer.comm,
    )


def test_calls_reslice(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike
):
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=4,
        chunk_start=3,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    block1 = dummy_dataset.make_block(0, 0, 2)
    block2 = dummy_dataset.make_block(0, 2, 2)

    writer.write_block(block1)
    writer.write_block(block2)

    reslice_mock = mocker.patch.object(DataSetStoreReader, "_reslice")
    d = writer._data
    writer.make_reader(new_slicing_dim=1)

    reslice_mock.assert_called_with(0, 1, d)


@pytest.mark.parametrize("file_based", [False, True])
def test_reslice_single_block_single_process(
    mocker: MockerFixture, dummy_dataset: DataSet, tmp_path: PathLike, file_based: bool
):
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=dummy_dataset.shape[0],
        chunk_start=0,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )
    if file_based:
        mocker.patch.object(writer, "_create_numpy_data", side_effect=MemoryError)

    writer.write_block(dummy_dataset.make_block(0, 0, writer.global_shape[0]))

    assert writer.is_file_based is file_based

    reader = writer.make_reader(new_slicing_dim=1)

    block = reader.read_block(1, 2)

    assert reader.slicing_dim == 1
    assert reader.global_shape == writer.global_shape
    assert reader.chunk_shape == writer.chunk_shape
    assert reader.chunk_index == (0, 0, 0)

    assert block.global_shape == reader.global_shape
    assert block.shape == (dummy_dataset.shape[0], 2, dummy_dataset.shape[2])
    assert block.chunk_index == (0, 1, 0)
    assert block.chunk_shape == reader.chunk_shape

    assert reader.is_file_based is file_based

    np.testing.assert_array_equal(block.data, dummy_dataset.data[:, 1:3, :])
    np.testing.assert_array_equal(block.flats, dummy_dataset.flats)
    np.testing.assert_array_equal(block.darks, dummy_dataset.darks)
    np.testing.assert_array_equal(block.angles, dummy_dataset.angles)


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
@pytest.mark.parametrize("out_of_memory_ranks", [
    [],
    [1], 
    [0, 1]
])
def test_reslice_single_block_multi_process(
    mocker: MockerFixture,
    dummy_dataset: DataSet,
    tmp_path: PathLike,
    out_of_memory_ranks: List[int],
):
    global_data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    dummy_dataset.data = global_data
    comm = MPI.COMM_WORLD

    GLOBAL_DATA_SHAPE = (10, 10, 10)
    if len(out_of_memory_ranks) > 0:
        # If using hdf5 file as storage underneath data store, then put global data inside a
        # `FullFileDataSet` object rather than a `DataSet` object
        dummy_dataset = FullFileDataSet(
            data=global_data,
            angles=np.ones((20,)),
            flats=3 * np.ones((5, 10, 10)),
            darks=2 * np.ones((5, 10, 10)),
            global_index=(0, 0, 0)
            if comm.rank == 0
            else (GLOBAL_DATA_SHAPE[0] // 2, 0, 0),
            chunk_shape=(5, 10, 10),
        )

    assert comm.size == 2

    writer = DataSetStoreWriter(
        full_size=dummy_dataset.shape[0],
        slicing_dim=0,
        other_dims=(dummy_dataset.shape[1], dummy_dataset.shape[2]),
        chunk_size=global_data.shape[0] // 2,
        chunk_start=0 if comm.rank == 0 else dummy_dataset.shape[0] // 2,
        comm=MPI.COMM_WORLD,
        temppath=tmp_path,
    )

    if comm.rank in out_of_memory_ranks:
        mocker.patch.object(writer, "_create_numpy_data", side_effect=MemoryError)

    if len(out_of_memory_ranks) == 0:
        reslice_mock = mocker.patch("httomo.data.dataset_store.reslice")

    if comm.rank == 0:
        writer.write_block(dummy_dataset.make_block(0, 0, writer.chunk_shape[0]))
        if len(out_of_memory_ranks) == 0:
            reslice_mock.return_value = (
                dummy_dataset.data[:, 0 : dummy_dataset.shape[1] // 2, :],
                2,
                0,
            )
    else:
        writer.write_block(dummy_dataset.make_block(0, 0, writer.chunk_shape[0]))
        if len(out_of_memory_ranks) == 0:
            reslice_mock.return_value = (
                dummy_dataset.data[:, dummy_dataset.shape[1] // 2 :, :],
                2,
                dummy_dataset.shape[1] // 2,
            )

    reader = writer.make_reader(new_slicing_dim=1)
    block = reader.read_block(1, 2)

    assert reader.slicing_dim == 1
    assert reader.global_shape == writer.global_shape
    assert reader.chunk_shape == (
        dummy_dataset.shape[0],
        dummy_dataset.shape[1] // 2,
        dummy_dataset.shape[2],
    )
    if comm.rank == 0:
        assert writer.chunk_index == (0, 0, 0)
        assert reader.chunk_index == (0, 0, 0)
    else:
        assert writer.chunk_index == (dummy_dataset.shape[0] // 2, 0, 0)
        assert reader.chunk_index == (0, dummy_dataset.shape[1] // 2, 0)

    assert block.global_shape == reader.global_shape
    assert block.shape == (dummy_dataset.shape[0], 2, dummy_dataset.shape[2])
    assert block.chunk_index == (0, 1, 0)
    assert block.chunk_shape == reader.chunk_shape

    if len(out_of_memory_ranks) == 0:
        reslice_mock.assert_called_once_with(ANY, 1, 2, ANY)
    if comm.rank == 0:
        np.testing.assert_array_equal(
            block.data,
            global_data[:, 1:3, :],
        )
    else:
        np.testing.assert_array_equal(
            block.data,
            global_data[
                :, dummy_dataset.shape[1] // 2 + 1 : dummy_dataset.shape[1] // 2 + 3, :
            ],
        )
    np.testing.assert_array_equal(block.flats, dummy_dataset.flats)
    np.testing.assert_array_equal(block.darks, dummy_dataset.darks)
    np.testing.assert_array_equal(block.angles, dummy_dataset.angles)
