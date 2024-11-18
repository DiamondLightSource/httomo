from pathlib import Path
from typing import Any, Tuple
from mpi4py import MPI
from pytest_mock import MockerFixture
from httomo.runner.output_ref import OutputRef
from httomo.ui_layer import UiLayer
from httomo import ui_layer
import pytest

from .testing_utils import make_test_method

# TODO: add files with invalid syntax


def test_can_read_cpu1(yaml_cpu_pipeline1):
    pipline_stage_config = ui_layer.yaml_loader(yaml_cpu_pipeline1)

    assert len(pipline_stage_config) == 7
    assert pipline_stage_config[0]["method"] == "standard_tomo"
    assert pipline_stage_config[0]["module_path"] == "httomo.data.hdf.loaders"
    assert pipline_stage_config[1]["method"] == "normalize"
    assert pipline_stage_config[1]["module_path"] == "tomopy.prep.normalize"
    assert pipline_stage_config[2]["method"] == "minus_log"
    assert pipline_stage_config[2]["module_path"] == "tomopy.prep.normalize"
    assert pipline_stage_config[3]["method"] == "find_center_vo"
    assert pipline_stage_config[3]["module_path"] == "tomopy.recon.rotation"
    assert pipline_stage_config[3]["side_outputs"] == {"cor": "centre_of_rotation"}
    assert pipline_stage_config[4]["method"] == "recon"
    assert pipline_stage_config[5]["method"] == "rescale_to_int"
    assert pipline_stage_config[5]["module_path"] == "httomolibgpu.misc.rescale"
    assert pipline_stage_config[4]["module_path"] == "tomopy.recon.algorithm"
    assert pipline_stage_config[6]["method"] == "save_to_images"
    assert pipline_stage_config[6]["module_path"] == "httomolib.misc.images"


def test_can_read_cpu2(yaml_cpu_pipeline2):
    pipline_stage_config = ui_layer.yaml_loader(yaml_cpu_pipeline2)

    assert len(pipline_stage_config) == 10
    assert pipline_stage_config[0]["method"] == "standard_tomo"
    assert pipline_stage_config[0]["module_path"] == "httomo.data.hdf.loaders"
    assert pipline_stage_config[0]["parameters"]["preview"] == {
        "detector_y": {"start": 30, "stop": 60}
    }
    assert pipline_stage_config[1]["method"] == "find_center_vo"
    assert pipline_stage_config[1]["module_path"] == "tomopy.recon.rotation"
    assert pipline_stage_config[1]["side_outputs"] == {"cor": "centre_of_rotation"}
    assert pipline_stage_config[2]["method"] == "remove_outlier"
    assert pipline_stage_config[2]["module_path"] == "tomopy.misc.corr"
    assert pipline_stage_config[3]["method"] == "normalize"
    assert pipline_stage_config[3]["module_path"] == "tomopy.prep.normalize"
    assert pipline_stage_config[4]["method"] == "minus_log"
    assert pipline_stage_config[4]["module_path"] == "tomopy.prep.normalize"
    assert pipline_stage_config[5]["method"] == "remove_stripe_fw"
    assert pipline_stage_config[5]["module_path"] == "tomopy.prep.stripe"
    assert pipline_stage_config[6]["method"] == "recon"
    assert pipline_stage_config[6]["module_path"] == "tomopy.recon.algorithm"
    assert pipline_stage_config[6]["save_result"] == True
    assert pipline_stage_config[7]["method"] == "median_filter"
    assert pipline_stage_config[7]["module_path"] == "tomopy.misc.corr"
    assert pipline_stage_config[8]["method"] == "rescale_to_int"
    assert pipline_stage_config[8]["module_path"] == "httomolibgpu.misc.rescale"
    assert pipline_stage_config[9]["method"] == "save_to_images"
    assert pipline_stage_config[9]["module_path"] == "httomolib.misc.images"


def test_can_read_cpu3(yaml_cpu_pipeline3):
    pipline_stage_config = ui_layer.yaml_loader(yaml_cpu_pipeline3)

    assert len(pipline_stage_config) == 9
    assert pipline_stage_config[0]["method"] == "standard_tomo"
    assert pipline_stage_config[0]["module_path"] == "httomo.data.hdf.loaders"
    assert pipline_stage_config[1]["method"] == "find_center_vo"
    assert pipline_stage_config[1]["module_path"] == "tomopy.recon.rotation"
    assert pipline_stage_config[1]["side_outputs"] == {"cor": "centre_of_rotation"}
    assert pipline_stage_config[2]["method"] == "remove_outlier"
    assert pipline_stage_config[2]["module_path"] == "tomopy.misc.corr"
    assert pipline_stage_config[3]["method"] == "normalize"
    assert pipline_stage_config[3]["module_path"] == "tomopy.prep.normalize"
    assert pipline_stage_config[4]["method"] == "minus_log"
    assert pipline_stage_config[4]["module_path"] == "tomopy.prep.normalize"
    assert pipline_stage_config[5]["method"] == "recon"
    assert pipline_stage_config[5]["module_path"] == "tomopy.recon.algorithm"
    assert pipline_stage_config[6]["method"] == "calculate_stats"
    assert pipline_stage_config[6]["module_path"] == "httomo.methods"
    assert pipline_stage_config[6]["side_outputs"] == {"glob_stats": "glob_stats"}
    assert pipline_stage_config[7]["method"] == "rescale_to_int"
    assert pipline_stage_config[7]["module_path"] == "httomolibgpu.misc.rescale"
    assert pipline_stage_config[8]["method"] == "save_to_images"
    assert pipline_stage_config[8]["module_path"] == "httomolib.misc.images"


def test_can_read_gpu1(yaml_gpu_pipeline1):
    pipline_stage_config = ui_layer.yaml_loader(yaml_gpu_pipeline1)

    assert len(pipline_stage_config) == 8
    assert pipline_stage_config[0]["method"] == "standard_tomo"
    assert pipline_stage_config[0]["module_path"] == "httomo.data.hdf.loaders"
    assert pipline_stage_config[1]["method"] == "find_center_vo"
    assert pipline_stage_config[1]["module_path"] == "httomolibgpu.recon.rotation"
    assert pipline_stage_config[1]["side_outputs"] == {"cor": "centre_of_rotation"}
    assert pipline_stage_config[2]["method"] == "remove_outlier"
    assert pipline_stage_config[2]["module_path"] == "httomolibgpu.misc.corr"
    assert pipline_stage_config[3]["method"] == "normalize"
    assert pipline_stage_config[3]["module_path"] == "httomolibgpu.prep.normalize"
    assert pipline_stage_config[4]["method"] == "remove_stripe_based_sorting"
    assert pipline_stage_config[4]["module_path"] == "httomolibgpu.prep.stripe"
    assert pipline_stage_config[5]["method"] == "FBP"
    assert pipline_stage_config[5]["module_path"] == "httomolibgpu.recon.algorithm"
    assert pipline_stage_config[6]["method"] == "rescale_to_int"
    assert pipline_stage_config[6]["module_path"] == "httomolibgpu.misc.rescale"
    assert pipline_stage_config[7]["method"] == "save_to_images"
    assert pipline_stage_config[7]["module_path"] == "httomolib.misc.images"


@pytest.mark.parametrize("file", ["does_not_exist.yaml"])
def test_uilayer_fails_with_nonexistant_file(file: str):
    comm = MPI.COMM_NULL
    with pytest.raises(FileNotFoundError):
        UiLayer(Path(file), Path("doesnt_matter"), comm=comm)


def test_pipeline_build_no_loader(yaml_cpu_pipeline1, standard_data):
    comm = MPI.COMM_NULL
    LayerUI = UiLayer(yaml_cpu_pipeline1, standard_data, comm=comm)
    del LayerUI.PipelineStageConfig[0]
    with pytest.raises(ValueError) as e:
        LayerUI.build_pipeline()

    assert "no loader" in str(e)


def test_pipeline_build_duplicate_id(yaml_cpu_pipeline1, standard_data):
    comm = MPI.COMM_NULL
    LayerUI = UiLayer(yaml_cpu_pipeline1, standard_data, comm=comm)
    LayerUI.PipelineStageConfig[1]["id"] = "testid"
    LayerUI.PipelineStageConfig[2]["id"] = "testid"
    with pytest.raises(ValueError) as e:
        LayerUI.build_pipeline()

    assert "duplicate id" in str(e)


def test_pipeline_build_cpu1(standard_data, yaml_cpu_pipeline1):
    """Testing OutputRef."""
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(yaml_cpu_pipeline1, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()

    assert len(pipeline) == 6
    methods = [
        "normalize",
        "minus_log",
        "find_center_vo",
        "recon",
        "rescale_to_int",
        "save_to_images",
    ]
    for i in range(5):
        assert pipeline[i].method_name == methods[i]
        if i != 2:
            assert pipeline[i].task_id == f"task_{i+1}"
        else:
            assert pipeline[i].task_id == "centering"

    ref = pipeline[3]["center"]
    assert isinstance(ref, OutputRef)
    assert ref.mapped_output_name == "centre_of_rotation"
    assert ref.method.method_name == "find_center_vo"


def test_pipeline_build_cpu2(standard_data, yaml_cpu_pipeline2):
    """Testing OutputRef."""
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(yaml_cpu_pipeline2, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()

    assert len(pipeline) == 9
    methods = [
        "find_center_vo",
        "remove_outlier",
        "normalize",
        "minus_log",
        "remove_stripe_fw",
        "recon",
        "median_filter",
        "rescale_to_int",
        "save_to_images",
    ]
    for i in range(8):
        assert pipeline[i].method_name == methods[i]
    ref = pipeline[5]["center"]
    assert isinstance(ref, OutputRef)
    assert ref.mapped_output_name == "centre_of_rotation"
    assert ref.method.method_name == "find_center_vo"
    assert ref.method.task_id == "centering"


def test_pipeline_build_cpu3(standard_data, yaml_cpu_pipeline3):
    """Testing OutputRef."""
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(yaml_cpu_pipeline3, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()

    assert len(pipeline) == 8
    methods = [
        "find_center_vo",
        "remove_outlier",
        "normalize",
        "minus_log",
        "recon",
        "calculate_stats",
        "rescale_to_int",
        "save_to_images",
    ]
    for i in range(8):
        assert pipeline[i].method_name == methods[i]
    ref = pipeline[6]["glob_stats"]
    assert isinstance(ref, OutputRef)
    assert ref.mapped_output_name == "glob_stats"
    assert ref.method.method_name == "calculate_stats"
    assert ref.method.task_id == "statistics"


@pytest.mark.parametrize(
    "pipeline_file, expected_sweep_vals",
    [
        ("samples/pipeline_template_examples/testing/sweep_manual.yaml", (3, 5)),
        (
            "samples/pipeline_template_examples/testing/sweep_range.yaml",
            tuple(range(3, 13)),
        ),
    ],
    ids=["manual", "range"],
)
def test_build_pipeline_with_param_sweeps(
    standard_data, pipeline_file: Path, expected_sweep_vals: Tuple[Any, ...]
):
    pipeline_file_path = Path(__file__).parent / pipeline_file
    ui_layer = UiLayer(
        tasks_file_path=pipeline_file_path,
        in_data_file_path=standard_data,
        comm=MPI.COMM_WORLD,
    )
    pipeline = ui_layer.build_pipeline()
    sweep_method_wrapper = pipeline[1]
    SWEEP_PARAM_NAME = "size"
    assert SWEEP_PARAM_NAME in sweep_method_wrapper.config_params.keys()
    assert sweep_method_wrapper.config_params[SWEEP_PARAM_NAME] == expected_sweep_vals


@pytest.mark.parametrize(
    "refvalue",
    [
        "no_ref",
        "as ${{mixed_with_other}}",
        "${{mixed_with_other}} asd",
        "${{}}",
        "${{with spaces}}",
        "${{inV-1%^&valid_chars}}",
    ],
)
def test_update_side_output_references_invalid(refvalue: str):
    parameters = {"refkey": refvalue}
    valid_refs = ui_layer.get_valid_ref_str(parameters)
    ui_layer.update_side_output_references(valid_refs, parameters, {})
    assert parameters == {"refkey": refvalue}  # not updated


def test_update_side_output_references_normal(mocker: MockerFixture):
    parameters = {"refkey": "${{testid.side_outputs.value}}"}
    valid_refs = ui_layer.get_valid_ref_str(parameters)
    ui_layer.update_side_output_references(
        valid_refs, parameters, dict(testid=make_test_method(mocker))
    )

    obj = parameters["refkey"]
    assert isinstance(obj, OutputRef)
    assert obj.mapped_output_name == "value"


def test_update_side_output_references_nosidestr():
    parameters = {"refkey": "${{testid.side_typo_outputs.value}}"}
    with pytest.raises(ValueError) as e:
        valid_refs = ui_layer.get_valid_ref_str(parameters)
        ui_layer.update_side_output_references(valid_refs, parameters, {})
        assert "side_outputs" in str(e)


def test_update_side_output_references_notfound(mocker: MockerFixture):
    parameters = {"refkey": "${{testid123.side_outputs.value}}"}
    with pytest.raises(ValueError) as e:
        valid_refs = ui_layer.get_valid_ref_str(parameters)
        ui_layer.update_side_output_references(
            valid_refs, parameters, dict(testid=make_test_method(mocker))
        )

    assert "could not find method referenced" in str(e)
