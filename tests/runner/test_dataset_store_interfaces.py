from httomo.runner.dataset import DataSet
from httomo.runner.dataset_store_interfaces import DummySink


def test_dummy_dataset_sink(dummy_dataset: DataSet):
    blk = dummy_dataset.make_block(0, 0, 3)
    dummy_sink = DummySink(0)
    dummy_sink.write_block(blk)
    
    assert dummy_sink.global_shape == dummy_dataset.global_shape
    assert dummy_sink.chunk_index == dummy_sink.chunk_index
    assert dummy_sink.chunk_shape == dummy_dataset.chunk_shape
    