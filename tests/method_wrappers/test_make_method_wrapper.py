from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.distortion_correction import DistortionCorrectionWrapper
from tests.testing_utils import make_mock_preview_config, make_mock_repo


def test_creates_distortion_correction_wrapper_and_passes_preview_through(
    mocker: MockerFixture,
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

    wrapper = make_method_wrapper(
        method_repository=make_mock_repo(mocker),
        module_path=MODULE_PATH,
        method_name=METHOD_NAME,
        comm=COMM,
        preview_config=make_mock_preview_config(mocker),
    )
    assert isinstance(wrapper, DistortionCorrectionWrapper)
    assert "shift_xy" in wrapper.config_params
    assert "step_xy" in wrapper.config_params
