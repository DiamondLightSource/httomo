from unittest.mock import call
import pytest
from pytest_mock import MockerFixture
from httomo.runner.block_split import BlockSplitter
from httomo.runner.dataset_store_interfaces import DataSetSource


def test_block_splitter_throws_with_slicingdim_2(mocker: MockerFixture):
    """This test should be removed if dim=2 is fully supported"""
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=(10, 10, 10), padding=(0, 0), slicing_dim=2
    )
    with pytest.raises(AssertionError):
        BlockSplitter(source, 100000000)


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_splitter_gives_full_when_fits(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=(0, 0), slicing_dim=slicing_dim
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
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=(0, 0), slicing_dim=slicing_dim
    )
    splitter = BlockSplitter(source, 100000000)

    with pytest.raises(IndexError):
        splitter[12]


@pytest.mark.parametrize("padding", [(0, 0), (1, 1)])
@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_splitter_splits_evenly(
    mocker: MockerFixture, slicing_dim: int, padding: tuple[int, int]
):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=padding, slicing_dim=slicing_dim
    )
    splitter = BlockSplitter(source, CHUNK_SHAPE[slicing_dim] // 2)

    assert (
        splitter.slices_per_block
        == CHUNK_SHAPE[slicing_dim] // 2 - padding[0] - padding[1]
    )

    if padding[0] == 0:
        assert len(splitter) == 2
    else:
        assert len(splitter) == 3


@pytest.mark.parametrize("padding", [(0, 0), (1, 1)])
@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_splitter_splits_odd(
    mocker: MockerFixture, slicing_dim: int, padding: tuple[int, int]
):
    CHUNK_SHAPE = (10, 7, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=padding, slicing_dim=slicing_dim
    )

    splitter = BlockSplitter(source, 3)

    assert splitter.slices_per_block == 3 - padding[0] - padding[1]
    if slicing_dim == 0:
        if padding[0] == 0:
            assert len(splitter) == 4
        else:
            assert len(splitter) == CHUNK_SHAPE[slicing_dim]
            assert len(splitter) == 10
    else:
        if padding[0] == 0:
            assert len(splitter) == 3
        else:
            assert len(splitter) == CHUNK_SHAPE[slicing_dim]
            assert len(splitter) == 7


@pytest.mark.parametrize("padding", [(0, 0), (1, 1)])
@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_gives_dataset_full(
    mocker: MockerFixture, slicing_dim: int, padding: tuple[int, int]
):
    CHUNK_SHAPE = (10, 7, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=padding, slicing_dim=slicing_dim
    )
    splitter = BlockSplitter(source, 100000000)

    splitter[0]

    source.read_block.assert_called_with(0, CHUNK_SHAPE[slicing_dim])


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_reads_blocks_even(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=(0, 0), slicing_dim=slicing_dim
    )

    max_slices = CHUNK_SHAPE[slicing_dim] // 2
    splitter = BlockSplitter(source, max_slices)

    splitter[0]
    splitter[1]

    source.read_block.assert_has_calls(
        [call(0, max_slices), call(max_slices, max_slices)]
    )


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_reads_blocks_even_padded(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    PADDING = (5, 5)
    TOTAL_PADDING = PADDING[0] + PADDING[1]
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=PADDING, slicing_dim=slicing_dim
    )

    max_slices = CHUNK_SHAPE[slicing_dim] // 2
    splitter = BlockSplitter(source, max_slices)

    splitter[0]
    splitter[1]
    splitter[2]

    source.read_block.assert_has_calls(
        [
            call(0, max_slices - TOTAL_PADDING),
            call(max_slices - TOTAL_PADDING, max_slices - TOTAL_PADDING),
            call(
                2 * (max_slices - TOTAL_PADDING),
                CHUNK_SHAPE[slicing_dim] - 2 * (max_slices - TOTAL_PADDING),
            ),
        ]
    )


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_reads_blocks_odd(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (5, 7, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=(0, 0), slicing_dim=slicing_dim
    )

    max_slices = 4
    splitter = BlockSplitter(source, max_slices)

    splitter[0]
    splitter[1]

    source.read_block.assert_has_calls(
        [call(0, max_slices), call(max_slices, CHUNK_SHAPE[slicing_dim] - max_slices)]
    )


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_reads_blocks_odd_padded(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (5, 7, 100)
    PADDING = (1, 1)
    TOTAL_PADDING = PADDING[0] + PADDING[1]
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=PADDING, slicing_dim=slicing_dim
    )

    max_slices = 4
    splitter = BlockSplitter(source, max_slices)

    splitter[0]
    splitter[1]
    splitter[2]

    if slicing_dim == 1:
        splitter[3]

    if slicing_dim == 0:
        source.read_block.assert_has_calls(
            [
                call(0, max_slices - TOTAL_PADDING),
                call(max_slices - TOTAL_PADDING, max_slices - TOTAL_PADDING),
                call(
                    2 * (max_slices - TOTAL_PADDING),
                    CHUNK_SHAPE[slicing_dim] - 2 * (max_slices - TOTAL_PADDING),
                ),
            ]
        )
    else:
        source.read_block.assert_has_calls(
            [
                call(0, max_slices - TOTAL_PADDING),
                call(max_slices - TOTAL_PADDING, max_slices - TOTAL_PADDING),
                call(2 * (max_slices - TOTAL_PADDING), max_slices - TOTAL_PADDING),
                call(
                    3 * (max_slices - TOTAL_PADDING),
                    CHUNK_SHAPE[slicing_dim] - 3 * (max_slices - TOTAL_PADDING),
                ),
            ]
        )


@pytest.mark.parametrize("slicing_dim", [0, 1], ids=["proj", "sino"])
def test_block_can_iterate(mocker: MockerFixture, slicing_dim: int):
    CHUNK_SHAPE = (100, 70, 100)
    source = mocker.create_autospec(
        DataSetSource, chunk_shape=CHUNK_SHAPE, padding=(0, 0), slicing_dim=slicing_dim
    )

    max_slices = CHUNK_SHAPE[slicing_dim] // 2
    splitter = BlockSplitter(source, max_slices)

    for i, _ in enumerate(splitter):
        assert i < 2

    source.read_block.assert_has_calls(
        [call(0, max_slices), call(max_slices, max_slices)]
    )
