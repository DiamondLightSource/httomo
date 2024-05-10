from unittest.mock import call
import pytest
from pytest_mock import MockerFixture
from httomo.runner.block_split import BlockSplitter
from httomo.runner.dataset_store_interfaces import DataSetSource


def test_block_splitter_throws_with_slicingdim_2(mocker: MockerFixture):
    """This test should be removed if dim=2 is fully supported"""
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=(10, 10, 10), slicing_dim=2
    )
    with pytest.raises(AssertionError):
        BlockSplitter(source, 100000000)


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_splitter_gives_full_when_fits(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )
    splitter = BlockSplitter(source, 100000000)

    assert splitter.slices_per_block == CHUNK_SHAPE[slicing_dim]
    assert len(splitter) == 1


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_splitter_raises_if_out_of_bounds(
    mocker: MockerFixture, slicing_dim: int
):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )
    splitter = BlockSplitter(source, 100000000)

    with pytest.raises(IndexError):
        splitter[12]


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_splitter_splits_evenly(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )
    splitter = BlockSplitter(source, CHUNK_SHAPE[slicing_dim] // 2)

    assert splitter.slices_per_block == CHUNK_SHAPE[slicing_dim] // 2
    assert len(splitter) == 2


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_splitter_splits_odd(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (10, 7, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )

    splitter = BlockSplitter(source, 3)

    assert splitter.slices_per_block == 3
    if slicing_dim == 0:
        assert len(splitter) == 4
    else:
        assert len(splitter) == 3


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_gives_dataset_full(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (10, 7, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )
    splitter = BlockSplitter(source, 100000000)

    splitter[0]

    source.read_block.assert_called_with(0, CHUNK_SHAPE[slicing_dim])


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_reads_blocks_even(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )

    max_slices = CHUNK_SHAPE[slicing_dim] // 2
    splitter = BlockSplitter(source, max_slices)

    splitter[0]
    splitter[1]

    source.read_block.assert_has_calls(
        [call(0, max_slices), call(max_slices, max_slices)]
    )


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_reads_blocks_odd(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (5, 7, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )

    max_slices = 4
    splitter = BlockSplitter(source, max_slices)

    splitter[0]
    splitter[1]

    source.read_block.assert_has_calls(
        [call(0, max_slices), call(max_slices, CHUNK_SHAPE[slicing_dim] - max_slices)]
    )


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_can_iterate(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, slicing_dim=slicing_dim
    )

    max_slices = CHUNK_SHAPE[slicing_dim] // 2
    splitter = BlockSplitter(source, max_slices)

    for i, _ in enumerate(splitter):
        assert i < 2

    source.read_block.assert_has_calls(
        [call(0, max_slices), call(max_slices, max_slices)]
    )

