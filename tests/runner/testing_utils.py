from httomo.runner.backend_wrapper import BackendWrapper
from httomo.runner.loader import LoaderInterface
from httomo.runner.methods_repository_interface import MethodRepository
from httomo.utils import Pattern

from pytest_mock import MockerFixture


def make_test_method(
    mocker: MockerFixture,
    gpu=False,
    pattern=Pattern.projection,
    method_name="testmethod",
    module_path="testpath",
    **kwargs,
) -> BackendWrapper:
    mock = mocker.create_autospec(
        BackendWrapper,
        instance=True,
        method_name=method_name,
        module_path=module_path,
        pattern=pattern,
        is_gpu=gpu,
        is_cpu=not gpu,
        config_params=kwargs,
        __getitem__=lambda _, k: kwargs[k],  # return kwargs value from dict access
    )

    return mock


def make_test_loader(
    mocker: MockerFixture, pattern: Pattern = Pattern.all, method_name="testloader"
) -> LoaderInterface:
    return mocker.create_autospec(
        LoaderInterface, instance=True, pattern=pattern, method_name=method_name
    )


def make_mock_repo(
    mocker: MockerFixture,
    pattern=Pattern.sinogram,
    output_dims_change=False,
    implementation="cpu",
    memory_gpu={"datasets": ["tomo"], "multipliers": [1.2], "methods": ["direct"]},
) -> MethodRepository:
    """Makes a mock MethodRepository that returns the given properties on any query"""
    mock_repo = mocker.MagicMock()
    mock_query = mocker.MagicMock()
    mocker.patch.object(mock_repo, "query", return_value=mock_query)
    mocker.patch.object(mock_query, "get_pattern", return_value=pattern)
    mocker.patch.object(
        mock_query, "get_output_dims_change", return_value=output_dims_change
    )
    mocker.patch.object(mock_query, "get_implementation", return_value=implementation)
    mocker.patch.object(mock_query, "get_memory_gpu_params", return_value=memory_gpu)
    return mock_repo


