import pytest

from httomo.method_wrappers.distortion_correction import DistortionCorrectionWrapper


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
