from typing import List, Literal, Optional
from httomo.preview import PreviewConfig
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.runner.loader import LoaderInterface
from httomo.runner.methods_repository_interface import (
    GpuMemoryRequirement,
    MethodRepository,
)

from httomo_backends.methods_database.query import Pattern

from pytest_mock import MockerFixture


def make_test_method(
    mocker: MockerFixture,
    gpu=False,
    pattern=Pattern.projection,
    method_name="testmethod",
    module_path="testpath",
    memory_gpu: Optional[GpuMemoryRequirement] = None,
    save_result=False,
    task_id: Optional[str] = None,
    padding: bool = False,
    sweep: bool = False,
    **kwargs,
) -> MethodWrapper:
    if task_id is None:
        task_id = f"task_{method_name}"
    mock = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        method_name=method_name,
        module_path=module_path,
        memory_gpu=memory_gpu,
        pattern=pattern,
        is_gpu=gpu,
        is_cpu=not gpu,
        save_result=save_result,
        task_id=task_id,
        config_params=kwargs,
        padding=padding,
        sweep=sweep,
        __getitem__=lambda _, k: kwargs[k],  # return kwargs value from dict access
    )

    return mock


def make_test_loader(
    mocker: MockerFixture,
    preview: Optional[PreviewConfig] = None,
    block: Optional[DataSetBlock] = None,
    pattern: Pattern = Pattern.all,
    method_name="testloader",
) -> LoaderInterface:
    interface: LoaderInterface = mocker.create_autospec(
        LoaderInterface,
        instance=True,
        preview=preview,
        pattern=pattern,
        method_name=method_name,
        reslice=False,
    )
    if block is not None:

        # NOTE: Even though the `padding` parameter is unused, this is needed in order to
        # replicate the signature of the `make_data_source()` method defined on the
        # `LoaderInterface` protocol
        def mock_make_data_source(padding) -> DataSetSource:
            ret = mocker.create_autospec(
                DataSetSource,
                preview=preview,
                global_shape=block.global_shape,
                dtype=block.data.dtype,
                chunk_shape=block.chunk_shape,
                chunk_index=block.chunk_index,
                slicing_dim=1 if interface.pattern == Pattern.sinogram else 0,
                aux_data=block.aux_data,
            )
            slicing_dim: Literal[0, 1, 2] = (
                1 if interface.pattern == Pattern.sinogram else 0
            )
            mocker.patch.object(
                ret,
                "read_block",
                side_effect=lambda start, length: DataSetBlock(
                    data=block.data[start : start + length, :, :],
                    aux_data=block.aux_data,
                    global_shape=block.global_shape,
                    chunk_shape=block.chunk_shape,
                    slicing_dim=slicing_dim,
                    block_start=start,
                    chunk_start=block.chunk_index[slicing_dim],
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
    memory_gpu: Optional[GpuMemoryRequirement] = GpuMemoryRequirement(
        multiplier=1.2, method="direct"
    ),
    swap_dims_on_output=False,
    save_result_default=False,
    padding=False,
) -> MethodRepository:
    """Makes a mock MethodRepository that returns the given properties on any query"""
    mock_repo = mocker.MagicMock()
    mock_query = mocker.MagicMock()
    mocker.patch.object(mock_repo, "query", return_value=mock_query)
    mocker.patch.object(mock_query, "get_pattern", return_value=pattern)
    mocker.patch.object(
        mock_query, "get_output_dims_change", return_value=output_dims_change
    )
    mocker.patch.object(
        mock_query, "swap_dims_on_output", return_value=swap_dims_on_output
    )
    mocker.patch.object(mock_query, "get_implementation", return_value=implementation)
    mocker.patch.object(mock_query, "get_memory_gpu_params", return_value=memory_gpu)
    mocker.patch.object(
        mock_query, "save_result_default", return_value=save_result_default
    )
    mocker.patch.object(mock_query, "padding", return_value=padding)
    return mock_repo


def make_mock_preview_config(mocker: MockerFixture) -> PreviewConfig:
    return mocker.create_autospec(PreviewConfig)
