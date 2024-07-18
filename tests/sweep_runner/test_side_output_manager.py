from pytest_mock import MockerFixture

from httomo.sweep_runner.side_output_manager import SideOutputManager
from tests.testing_utils import make_test_method


def test_no_labels_after_creation():
    side_output_manager = SideOutputManager()
    assert len(side_output_manager.labels) == 0


def test_append_side_outputs():
    side_outputs = {"centre_of_rotation": 100.5}
    side_output_manager = SideOutputManager()
    side_output_manager.append(side_outputs)
    assert side_output_manager.labels == ["centre_of_rotation"]
    assert (
        side_output_manager.get("centre_of_rotation")
        == side_outputs["centre_of_rotation"]
    )


def test_update_method_with_side_output(mocker: MockerFixture):
    # Define single side output and create side output manager
    SIDE_OUTPUT_LABEL = "centre_of_rotation"
    side_outputs = {SIDE_OUTPUT_LABEL: 100.5}
    side_output_manager = SideOutputManager()
    side_output_manager.append(side_outputs)

    # Define mock method wrapper which requires the side output stored in the side output
    # manager
    method = make_test_method(mocker)
    mocker.patch.object(method, "parameters", [SIDE_OUTPUT_LABEL])
    setitem_mock = mocker.patch.object(method, "__setitem__")

    # Update method wrapper param to contain value of side output and assert that the wrapper's
    # relevant param was attempted to be updated
    side_output_manager.update_params(method)
    setitem_mock.assert_called_once_with(
        SIDE_OUTPUT_LABEL, side_outputs[SIDE_OUTPUT_LABEL]
    )
