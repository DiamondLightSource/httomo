from pathlib import Path
import pytest
from pytest_mock import MockerFixture
from httomo.runner.loader import LoaderInterface
from httomo.runner.pipeline import Pipeline
from httomo.transform_layer import TransformLayer
from .testing_utils import make_mock_repo, make_test_method
from mpi4py import MPI


def test_insert_save_methods(mocker: MockerFixture, tmp_path: Path):
    comm = MPI.COMM_SELF
    repo = make_mock_repo(mocker)
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[
            make_test_method(mocker, method_name="m1", save_result=True, task_id="t1"),
            make_test_method(mocker, method_name="m2", save_result=False),
        ],
    )
    trans = TransformLayer(comm, repo=repo, save_all=False, out_dir=tmp_path)
    pipeline = trans.insert_save_methods(pipeline)

    assert len(pipeline) == 3
    assert pipeline[1].method_name == "save_intermediate_data"
    assert pipeline[1].task_id == "save_t1"


def test_insert_save_methods_save_all(mocker: MockerFixture, tmp_path: Path):
    comm = MPI.COMM_SELF
    repo = make_mock_repo(mocker)
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[
            make_test_method(mocker, method_name="m1", save_result=False, task_id="t1"),
            make_test_method(mocker, method_name="m2", save_result=False, task_id="t2"),
        ],
    )
    trans = TransformLayer(comm, repo=repo, save_all=True, out_dir=tmp_path)
    pipeline = trans.insert_save_methods(pipeline)

    assert len(pipeline) == 4
    assert pipeline[1].method_name == "save_intermediate_data"
    assert pipeline[1].task_id == "save_t1"
    assert pipeline[3].method_name == "save_intermediate_data"
    assert pipeline[3].task_id == "save_t2"


@pytest.mark.parametrize("method_name", ["save_to_images", "find_center_vo"])
def test_insert_save_methods_does_not_save_for_save_to_images(
    mocker: MockerFixture, method_name: str, tmp_path: Path
):
    comm = MPI.COMM_SELF
    repo = make_mock_repo(mocker)
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[
            make_test_method(mocker, method_name="m1", save_result=False, task_id="t1"),
            make_test_method(
                mocker, method_name=method_name, save_result=True, task_id="t2"
            ),
        ],
    )
    trans = TransformLayer(comm, repo=repo, save_all=False, out_dir=tmp_path)
    pipeline = trans.insert_save_methods(pipeline)

    assert len(pipeline) == 2


def test_insert_save_methods_does_nothing_if_no_save(
    mocker: MockerFixture, tmp_path: Path
):
    comm = MPI.COMM_SELF
    repo = make_mock_repo(mocker)
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[
            make_test_method(mocker, method_name="m1", save_result=False, task_id="t1"),
            make_test_method(mocker, method_name="m2", save_result=False, task_id="t2"),
        ],
    )
    trans = TransformLayer(comm, repo=repo, save_all=False, out_dir=tmp_path)
    pipeline = trans.insert_save_methods(pipeline)

    assert len(pipeline) == 2


def test_transform_calls_insert_save_methods(mocker: MockerFixture, tmp_path: Path):
    comm = MPI.COMM_SELF
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[],
    )
    trans = TransformLayer(comm=comm, save_all=False, out_dir=tmp_path)
    save_mock = mocker.patch.object(trans, "insert_save_methods", return_value=pipeline)
    trans.transform(pipeline)

    save_mock.assert_called_once()


def test_insert_data_reducer(mocker: MockerFixture, tmp_path: Path):
    comm = MPI.COMM_SELF
    repo = make_mock_repo(mocker)
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[
            make_test_method(
                mocker, method_name="remove_outlier", save_result=False, task_id="t1"
            ),
            make_test_method(mocker, method_name="normalize", save_result=False),
        ],
    )
    trans = TransformLayer(comm, repo=repo, save_all=False, out_dir=tmp_path)
    pipeline = trans.insert_data_reducer(pipeline)

    assert len(pipeline) == 3
    assert pipeline[0].method_name == "data_reducer"
    assert pipeline[0].task_id == "reducer_0"


def test_insert_image_save_after_sweep(mocker: MockerFixture, tmp_path: Path):
    comm = MPI.COMM_SELF
    repo = make_mock_repo(mocker)
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[
            make_test_method(
                mocker,
                method_name="normalize",
                module_path="httomolibgpu.prep.normalize",
                save_result=False,
                task_id="t1",
            ),
            make_test_method(
                mocker,
                method_name="remove_outlier",
                module_path="httomolibgpu.misc.corr",
                save_result=False,
                task_id="t2",
            ),
            make_test_method(
                mocker,
                method_name="paganin_filter_tomopy",
                module_path="httomolibgpu.prep.phase",
                save_result=False,
                sweep=True,
                task_id="t3",
            ),
        ],
    )
    trans = TransformLayer(comm, repo=repo, save_all=False, out_dir=tmp_path)
    pipeline = trans.insert_save_images_after_sweep(pipeline)

    assert len(pipeline) == 4
    assert pipeline[3].method_name == "save_to_images"
    assert pipeline[3].task_id == "saveimage_sweep_t3"
    assert (
        pipeline[3].config_params["subfolder_name"]
        == "images_sweep_paganin_filter_tomopy"
    )
    assert pipeline[3].config_params["axis"] == 1


def test_insert_image_save_after_sweep2(mocker: MockerFixture, tmp_path: Path):
    comm = MPI.COMM_SELF
    repo = make_mock_repo(mocker)
    loader = mocker.create_autospec(
        LoaderInterface,
        instance=True,
    )
    pipeline = Pipeline(
        loader=loader,
        methods=[
            make_test_method(
                mocker,
                method_name="normalize",
                module_path="httomolibgpu.prep.normalize",
                save_result=False,
                task_id="t1",
            ),
            make_test_method(
                mocker,
                method_name="remove_outlier",
                module_path="httomolibgpu.misc.corr",
                save_result=False,
                task_id="t2",
            ),
            make_test_method(
                mocker,
                method_name="paganin_filter_tomopy",
                module_path="httomolibgpu.prep.phase",
                save_result=False,
                sweep=True,
                task_id="t3",
            ),
            make_test_method(
                mocker,
                method_name="FBP",
                module_path="httomolibgpu.recon.algorithm",
                save_result=False,
                task_id="t4",
            ),
        ],
    )
    trans = TransformLayer(comm, repo=repo, save_all=False, out_dir=tmp_path)
    pipeline = trans.insert_save_images_after_sweep(pipeline)

    assert len(pipeline) == 6
    assert pipeline[3].method_name == "save_to_images"
    assert pipeline[3].task_id == "saveimage_sweep_t3"
    assert (
        pipeline[3].config_params["subfolder_name"]
        == "images_sweep_paganin_filter_tomopy"
    )

    assert pipeline[5].method_name == "save_to_images"
    assert pipeline[5].task_id == "saveimage_sweep_t4"
    assert pipeline[5].config_params["subfolder_name"] == "images_sweep_FBP"
    assert pipeline[5].config_params["axis"] == 1
