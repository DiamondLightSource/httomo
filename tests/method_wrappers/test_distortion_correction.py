import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.method_wrappers.distortion_correction import DistortionCorrectionWrapper
from httomo.preview import PreviewConfig, PreviewDimConfig
from tests.testing_utils import make_mock_repo


@pytest.mark.parametrize(
    "method_name, expected_result",
    [("distortion_correction", True), ("other_method", False)],
    ids=["should-select", "shouldn't-select"],
)
def test_class_only_selected_for_methods_with_distortion_correction_in_name(
    method_name: str, expected_result: bool
):
    assert (
        DistortionCorrectionWrapper.should_select_this_class(
            "dummy.module.path", method_name
        )
        is expected_result
    )


def test_requires_preview_is_true():
    assert DistortionCorrectionWrapper.requires_preview() is True


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
            detector_y=PreviewDimConfig(start=5, stop=123),
            detector_x=PreviewDimConfig(start=0, stop=160),
        ),
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=0, stop=128),
            detector_x=PreviewDimConfig(start=5, stop=155),
        ),
    ],
    ids=["no_cropping", "crop_det_y_both_ends", "crop_det_x_both_ends"],
)
def test_sets_shiftxy_and_stepxy_params_correctly(
    preview_config: PreviewConfig, mocker: MockerFixture
):
    MODULE_PATH = "dummy.module.path"
    METHOD_NAME = "distortion_correction_dummy"
    COMM = MPI.COMM_WORLD

    # Patch method function import that occurs when the wrapper object is created, to instead
    # import the below dummy method function
    class FakeModule:
        def distortion_correction_dummy(shift_xy, step_xy):  # type: ignore
            return shift_xy + step_xy

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    wrapper = DistortionCorrectionWrapper(
        method_repository=make_mock_repo(mocker),
        module_path=MODULE_PATH,
        method_name=METHOD_NAME,
        comm=COMM,
        preview_config=preview_config,
    )

    # Check that the shift and step parameter values are present in the wrapper's parameter
    # config, and have the expected values based on the preview config
    expected_shift_values = [
        preview_config.detector_x.start,
        preview_config.detector_y.start,
    ]
    expected_step_values = [1, 1]

    SHIFT_PARAM_NAME = "shift_xy"
    STEP_PARAM_NAME = "step_xy"
    assert SHIFT_PARAM_NAME in wrapper.config_params
    assert wrapper.config_params[SHIFT_PARAM_NAME] == expected_shift_values
    assert STEP_PARAM_NAME in wrapper.config_params
    assert wrapper.config_params[STEP_PARAM_NAME] == expected_step_values
