import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.method_wrappers.seam_blender import SeamBlenderWrapper
from httomo.preview import PreviewConfig, PreviewDimConfig
from tests.testing_utils import make_mock_repo


@pytest.mark.parametrize(
    "method_name, expected_result",
    [("seam_blend", True), ("other_method", False)],
    ids=["should-select", "shouldn't-select"],
)
def test_class_only_selected_for_methods_with_seam_blend_in_name(
    method_name: str, expected_result: bool
):
    assert (
        SeamBlenderWrapper.should_select_this_class(
            "dummy.module.path", method_name
        )
        is expected_result
    )


def test_requires_preview_is_true():
    assert SeamBlenderWrapper.requires_preview() is True


@pytest.mark.parametrize(
    "preview_config",
    [
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=0, stop=128),
            detector_x=PreviewDimConfig(start=0, stop=160),
        ),
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=0, stop=128),
            detector_x=PreviewDimConfig(start=5, stop=155),
        ),
    ],
    ids=["no_cropping", "crop_det_x_both_ends"],
)
def test_sets_shiftx_params_correctly(
    preview_config: PreviewConfig, mocker: MockerFixture
):
    MODULE_PATH = "dummy.module.path"
    METHOD_NAME = "seam_blend_dummy"
    COMM = MPI.COMM_WORLD

    # Patch method function import that occurs when the wrapper object is created, to instead
    # import the below dummy method function
    class FakeModule:
        def seam_blend_dummy(shift_seam_index):  # type: ignore
            return shift_seam_index

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    wrapper = SeamBlenderWrapper(
        method_repository=make_mock_repo(mocker),
        module_path=MODULE_PATH,
        method_name=METHOD_NAME,
        comm=COMM,
        preview_config=preview_config,
    )

    expected_shift_values = preview_config.detector_x.start

    SHIFT_PARAM_NAME = "shift_seam_index"
    assert SHIFT_PARAM_NAME in wrapper.config_params
    assert wrapper.config_params[SHIFT_PARAM_NAME] == expected_shift_values
