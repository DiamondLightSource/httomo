from typing import List, Optional
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.dataset import DataSet
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.runner.loader import LoaderInterface
from httomo.runner.methods_repository_interface import (
    GpuMemoryRequirement,
    MethodRepository,
)
from httomo.utils import Pattern

from pytest_mock import MockerFixture


def make_test_method(
    mocker: MockerFixture,
    gpu=False,
    pattern=Pattern.projection,
    method_name="testmethod",
    module_path="testpath",
    save_result=False,
    task_id:Optional[str]=None,
    **kwargs,
) -> MethodWrapper:
    if task_id is None:
        task_id = f"task_{method_name}"
    mock = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        method_name=method_name,
        module_path=module_path,
        pattern=pattern,
        is_gpu=gpu,
        is_cpu=not gpu,
        save_result=save_result,
        task_id=task_id,
        config_params=kwargs,
        __getitem__=lambda _, k: kwargs[k],  # return kwargs value from dict access
    )

    return mock


def make_test_loader(
    mocker: MockerFixture,
    dataset: Optional[DataSet] = None,
    pattern: Pattern = Pattern.all,
    method_name="testloader",
) -> LoaderInterface:
    interface: LoaderInterface = mocker.create_autospec(
        LoaderInterface,
        instance=True,
        pattern=pattern,
        method_name=method_name,
        reslice=False,
    )
    if dataset is not None:

        def mock_make_data_source() -> DataSetSource:
            ret = mocker.create_autospec(
                DataSetSource,
                global_shape=dataset.global_shape,
                dtype=dataset.data.dtype,
                chunk_shape=dataset.chunk_shape,
                chunk_index=dataset.chunk_index,
                slicing_dim=1 if interface.pattern == Pattern.sinogram else 0,
                darks=dataset.darks,
                flats=dataset.flats,
            )
            mocker.patch.object(
                ret,
                "read_block",
                side_effect=lambda start, length: dataset.make_block(
                    1 if interface.pattern == Pattern.sinogram else 0, start, length
                ),
            )
            return ret

        mocker.patch.object(
            interface,
            "make_data_source",
            side_effect=mock_make_data_source,
        )
    return interface


def make_mock_repo(
    mocker: MockerFixture,
    pattern=Pattern.sinogram,
    output_dims_change: bool = False,
    implementation: str = "cpu",
    memory_gpu: List[GpuMemoryRequirement] = [
        GpuMemoryRequirement(dataset="tomo", multiplier=1.2, method="direct")
    ],
    swap_dims_on_output=False,
    save_result_default=False
) -> MethodRepository:
    """Makes a mock MethodRepository that returns the given properties on any query"""
    mock_repo = mocker.MagicMock()
    mock_query = mocker.MagicMock()
    mocker.patch.object(mock_repo, "query", return_value=mock_query)
    mocker.patch.object(mock_query, "get_pattern", return_value=pattern)
    mocker.patch.object(
        mock_query, "get_output_dims_change", return_value=output_dims_change
    )
    mocker.patch.object(mock_query, "swap_dims_on_output", return_value=swap_dims_on_output)
    mocker.patch.object(mock_query, "get_implementation", return_value=implementation)
    mocker.patch.object(mock_query, "get_memory_gpu_params", return_value=memory_gpu)
    mocker.patch.object(mock_query, "save_result_default", return_value=save_result_default)
    return mock_repo
