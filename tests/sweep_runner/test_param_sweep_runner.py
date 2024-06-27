import pytest

from httomo.sweep_runner.param_sweep_runner import ParamSweepRunner


def test_without_prepare_block_property_raises_error():
    runner = ParamSweepRunner()
    with pytest.raises(ValueError) as e:
        runner.block
    assert "Block from input data has not yet been loaded" in str(e)
