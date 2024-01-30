import httomo
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.images import ImagesWrapper
from httomo.runner.dataset import DataSet
from httomo.utils import gpu_enabled, xp
from ..testing_utils import make_mock_repo

import pytest
import numpy as np
from mpi4py import MPI
from pytest_mock import MockerFixture


def test_save_to_images(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def save_to_images(data, out_dir, axis, file_format, offset):
            np.testing.assert_array_equal(data, 1)
            assert out_dir == httomo.globals.run_out_dir
            assert offset == 2
            assert axis == 1
            assert file_format == "tif"

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="cpu"),
        "mocked_module_path.images",
        "save_to_images",
        MPI.COMM_WORLD,
        axis=1,
        file_format="tif",
    )
    assert isinstance(wrp, ImagesWrapper)
    
    # images is a sinogram method - slice in that dim
    block = dummy_dataset.make_block(1, 2)
    newblock = wrp.execute(block)

    assert newblock == block
    
def test_save_to_images_with_comm_rank_raises(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def save_to_images(data, out_dir, comm_rank, axis, file_format):
            pass 
        
    mocker.patch("importlib.import_module", return_value=FakeModule)
    with pytest.raises(ValueError) as e:
        make_method_wrapper(
            make_mock_repo(mocker, implementation="cpu"),
            "mocked_module_path.images",
            "save_to_images",
            MPI.COMM_WORLD,
            axis=1,
            file_format="tif",
        )
        
    assert "comm_rank" in str(e)


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.cupy
def test_images_leaves_gpudata(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def save_to_images(data, out_dir, offset):
            assert getattr(data, "device", None) is None  # make sure it's on CPU

    mocker.patch("importlib.import_module", return_value=FakeModule)
    wrp = make_method_wrapper(
        make_mock_repo(mocker),
        "mocked_module_path.images",
        "save_to_images",
        MPI.COMM_WORLD,
    )
    with xp.cuda.Device(0):
        dummy_dataset.to_gpu()
        new_dataset = wrp.execute(dummy_dataset.make_block(0))

        assert new_dataset.is_gpu is True