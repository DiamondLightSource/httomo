from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import DummySink


def test_dummy_dataset_sink(dummy_block: DataSetBlock):
    dummy_sink = DummySink(0)
    dummy_sink.write_block(dummy_block)

    assert dummy_sink.global_shape == dummy_block.global_shape
    assert dummy_sink.global_index == dummy_block.chunk_index
    assert dummy_sink.chunk_shape == dummy_block.chunk_shape
