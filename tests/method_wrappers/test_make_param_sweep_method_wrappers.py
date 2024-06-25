from ..testing_utils import make_mock_repo

from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.method_wrappers import make_param_sweep_method_wrappers


def test_make_param_sweep_method_wrappers(mocker: MockerFixture):
    NO_OF_SWEEPS = 5
    PARAM_1_VAL = 5
    PARAM_2_VAL = 0.1
    PARAMS = {"param_1": PARAM_1_VAL, "param_2": PARAM_2_VAL}
    PARAM_NAME_TO_SWEEP_OVER = "param_3"
    SWEEP_VALUES = list(range(0, 10 * NO_OF_SWEEPS, 10))
    MODULE_PATH = "mocked_module_path.corr"
    METHOD_NAME = "mock_method_name"
    COMM = MPI.COMM_WORLD

    # Define a fake module and fake method function within that module to be imported during
    # the creation of the method wrapper object
    class FakeModule:
        def mock_method_name(
            param_1: int, param_2: float, param_3: int  #  type: ignore
        ):
            return param_1 * param_2 + param_3

    mocker.patch("importlib.import_module", return_value=FakeModule)
    param_sweep_wrappers = make_param_sweep_method_wrappers(
        method_repository=make_mock_repo(mocker),
        module_path=MODULE_PATH,
        method_name=METHOD_NAME,
        comm=COMM,
        parameters=PARAMS,
        parameter_name=PARAM_NAME_TO_SWEEP_OVER,
        sweep_values=SWEEP_VALUES,
    )

    for wrp, val in zip(param_sweep_wrappers, SWEEP_VALUES):
        assert wrp.config_params["param_1"] == PARAM_1_VAL
        assert wrp.config_params["param_2"] == PARAM_2_VAL
        assert wrp.config_params[PARAM_NAME_TO_SWEEP_OVER] == val
