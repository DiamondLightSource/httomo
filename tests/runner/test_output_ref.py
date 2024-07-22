import pytest
from pytest_mock import MockerFixture

from httomo.runner.output_ref import OutputRef

from ..testing_utils import make_test_method


def test_gets_mapped_value(mocker: MockerFixture):
    m = make_test_method(mocker)
    mocker.patch.object(m, "get_side_output", return_value={"output": 42})
    ref = OutputRef(m, "output")
    assert ref.value == 42


def test_throws_if_not_mapped(mocker: MockerFixture):
    m = make_test_method(mocker)
    mocker.patch.object(
        m, "get_side_output", return_value={"output": 42, "xxx": [1, 2, 3]}
    )
    ref = OutputRef(m, "unknown_output")
    with pytest.raises(ValueError) as e:
        ref.value
    assert "not found in" in str(e)
    assert "unknown_output" in str(e)
    assert "are known: output, xxx"
