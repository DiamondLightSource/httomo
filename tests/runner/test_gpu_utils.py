import pytest
from mpi4py import MPI
from pytest_mock import MockerFixture

from httomo.runner.gpu_utils import get_available_gpu_memory, get_gpu_id
from httomo.utils import xp, gpu_enabled


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_get_available_memory():
    mem = get_available_gpu_memory(10.0)

    assert mem > 0
    assert mem <= 0.9 * xp.cuda.Device(get_gpu_id()).mem_info[0]


def test_get_available_memory_cpu(mocker: MockerFixture):
    if gpu_enabled:
        # this function is called in the implementation try block -
        # we trigger an import error here to simulate cupy not being there
        mocker.patch("cupy.cuda.Device", side_effect=ImportError("can't use cupy"))
    # greater 10GB
    assert get_available_gpu_memory() > 10e9


@pytest.mark.parametrize(
    "nodes, gpus_per_node, global_rank, local_rank",
    [
        (1, 4, 0, 0),
        (1, 4, 2, 2),
        (1, 2, 1, 1),
        (2, 4, 0, 0),
        (2, 4, 6, 2),
        (2, 2, 1, 1),
        (2, 2, 3, 1),
    ],
    ids=[
        "single-node-4gpus-per-node-rank-0",
        "single-node-4gpus-per-node-rank-2",
        "single-node-2gpus-per-node-rank-1",
        "multi-node-4gpus-per-node-rank-0",
        "multi-node-4gpus-per-node-rank-6",
        "multi-node-2gpus-per-node-rank-1",
        "multi-node-2gpus-per-node-rank-3",
    ],
)
def test_get_gpu_id(
    mocker: MockerFixture,
    nodes: int,
    gpus_per_node: int,
    global_rank: int,
    local_rank: int,
):
    # Mock global communicator object to reflect the number of nodes and gpus per node
    mock_global_comm = mocker.create_autospec(MPI.Comm)
    mock_global_comm.size = nodes * gpus_per_node
    mock_global_comm.rank = global_rank
    # Mock "local" communicator object (the comm local to a single node) to reflect the number
    # of gpus per node
    mock_local_comm = mocker.create_autospec(MPI.Comm)
    mock_local_comm.size = gpus_per_node
    mock_local_comm.rank = local_rank
    # Patch mock global comm to return mock local comm when the global comm is "split"
    mocker.patch.object(mock_global_comm, "Split_type", return_value=mock_local_comm)

    # Patch `httomo.runner.gpu_utils` import of `MPI.COMM_WORLD` to be the mock global
    # communicator object defined
    mocker.patch("httomo.runner.gpu_utils.MPI.COMM_WORLD", mock_global_comm)
    # Patch `cupy.cuda.runtime.getDeviceCount()` to return tbe desired number of GPUs that the
    # parametrisation of the test is wanting to check
    mocker.patch("cupy.cuda.runtime.getDeviceCount", return_value=gpus_per_node)

    # Check that the GPU ID returned is the expected GPU ID
    gpu_id = get_gpu_id()
    assert gpu_id == local_rank
