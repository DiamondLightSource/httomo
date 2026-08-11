# NOTE: those tests have path integrated that are compatible with running jobs in Jenkins at DLS infrastructure.
import subprocess
from typing import Callable
import pytest
import os
from .conftest import (
    change_value_parameters_method_pipeline,
    check_tif,
    compare_tif,
    calculate_gt_residual,
)


@pytest.mark.full_data
def test_pipe_tomopy_tomobank_preview(
    get_files: Callable,
    cmd,
    tomobank_00088,
    tomopy_tomobank,
    tomobank00088_tomopy_npz,
    output_folder,
):
    change_value_parameters_method_pipeline(
        tomopy_tomobank,
        method=[
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "find_center_vo",
        ],
        key=[
            "preview",
            "data_path",
            "image_key_path",
            "rotation_angles",
            "darks",
            "flats",
            "ind",
        ],
        value=[
            {"detector_y": {"start": 500, "stop": 510}},
            "/exchange/data",
            None,
            {
                "user_defined": {
                    "start_angle": 0,
                    "stop_angle": 179.876,
                    "angles_total": 1500,
                }
            },
            {
                "file": "input_data",
                "image_key_path": None,
                "data_path": "/exchange/data_dark",
            },
            {
                "file": "input_data",
                "image_key_path": None,
                "data_path": "/exchange/data_white",
            },
            "mid",
        ],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, tomobank_00088)
    cmd.insert(7, tomopy_tomobank)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = tomobank00088_tomopy_npz["data"]
    axis_slice = tomobank00088_tomopy_npz["axis_slice"]

    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="tomopy",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )

    assert res_norm < 0.1


@pytest.mark.full_data
def test_pipe_FBP3d_tomobar_tomobank_preview(
    get_files: Callable,
    cmd,
    tomobank_00088,
    FBP3d_tomobar_tomobank,
    tomobank00088_FBP3d_tomobar,
    output_folder,
):
    change_value_parameters_method_pipeline(
        FBP3d_tomobar_tomobank,
        method=[
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "find_center_vo",
        ],
        key=[
            "preview",
            "data_path",
            "image_key_path",
            "rotation_angles",
            "darks",
            "flats",
            "ind",
        ],
        value=[
            {"detector_y": {"start": 500, "stop": 510}},
            "/exchange/data",
            None,
            {
                "user_defined": {
                    "start_angle": 0,
                    "stop_angle": 179.876,
                    "angles_total": 1500,
                }
            },
            {
                "file": "input_data",
                "image_key_path": None,
                "data_path": "/exchange/data_dark",
            },
            {
                "file": "input_data",
                "image_key_path": None,
                "data_path": "/exchange/data_white",
            },
            "mid",
        ],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, tomobank_00088)
    cmd.insert(7, FBP3d_tomobar_tomobank)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = tomobank00088_FBP3d_tomobar["data"]
    axis_slice = tomobank00088_FBP3d_tomobar["axis_slice"]

    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="FBP3d_tomobar",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )

    assert res_norm < 1e-4


@pytest.mark.full_data
def test_pipe_FBP3d_tomobar_k11_38731_in_disk(
    get_files: Callable,
    cmd,
    diad_k11_38731,
    FBP3d_tomobar_noimagesaving,
    FBP3d_tomobar_k11_38731_npz,
    output_folder,
):
    change_value_parameters_method_pipeline(
        FBP3d_tomobar_noimagesaving,
        method=[
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
        ],
        key=[
            "data_path",
            "image_key_path",
            "rotation_angles",
        ],
        value=[
            "/entry/imaging/data",
            "/entry/instrument/imaging/image_key",
            {"data_path": "/entry/imaging_sum/gts_cs_theta"},
        ],
    )

    # NOTE that the intermediate file with file-based processing will be saved to /scratch/jenkins_agent/workspace/
    cmd.pop(4)  #: don't save all
    cmd.insert(5, diad_k11_38731)
    cmd.insert(7, FBP3d_tomobar_noimagesaving)
    cmd.insert(8, output_folder)
    cmd.insert(9, "--max-memory")
    cmd.insert(10, "5G")
    cmd.insert(11, "--reslice-dir")
    cmd.insert(12, "/scratch/jenkins_agent/workspace/")

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP3d_tomobar_k11_38731_npz["data"]
    axis_slice = FBP3d_tomobar_k11_38731_npz["axis_slice"]
    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="FBP3d_tomobar",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )
    assert res_norm < 1e-6


# ########################################################################
@pytest.mark.full_data
def test_pipe_FBP3d_tomobar_i12_119647_preview(
    get_files: Callable,
    cmd,
    i12_119647,
    FBP3d_tomobar,
    FBP3d_tomobar_i12_119647_npz,
    output_folder,
):

    change_value_parameters_method_pipeline(
        FBP3d_tomobar,
        method=[
            "standard_tomo",
            "remove_all_stripe",
        ],
        key=[
            "preview",
            "normalize",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
            True,
        ],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, i12_119647)
    cmd.insert(7, FBP3d_tomobar)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP3d_tomobar_i12_119647_npz["data"]
    axis_slice = FBP3d_tomobar_i12_119647_npz["axis_slice"]
    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="FBP3d_tomobar",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )
    assert res_norm < 1e-4


# ########################################################################


@pytest.mark.full_data
def test_pipe_LPRec3d_tomobar_i12_119647_preview(
    get_files: Callable,
    cmd,
    i12_119647,
    LPRec3d_tomobar,
    LPRec3d_tomobar_i12_119647_npz,
    output_folder,
):

    change_value_parameters_method_pipeline(
        LPRec3d_tomobar,
        method=[
            "standard_tomo",
            "remove_all_stripe",
        ],
        key=[
            "preview",
            "normalize",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
            True,
        ],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, i12_119647)
    cmd.insert(7, LPRec3d_tomobar)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = LPRec3d_tomobar_i12_119647_npz["data"]
    axis_slice = LPRec3d_tomobar_i12_119647_npz["axis_slice"]
    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="LPRec3d_tomobar",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )

    assert res_norm < 1e-4


# ########################################################################


@pytest.mark.full_data
def test_pipe_FBP2d_astra_i12_119647_preview(
    get_files: Callable,
    cmd,
    i12_119647,
    FBP2d_astra,
    FBP2d_astra_i12_119647_npz,
    output_folder,
):

    change_value_parameters_method_pipeline(
        FBP2d_astra,
        method=[
            "standard_tomo",
        ],
        key=[
            "preview",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
        ],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, i12_119647)
    cmd.insert(7, FBP2d_astra)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP2d_astra_i12_119647_npz["data"]
    axis_slice = FBP2d_astra_i12_119647_npz["axis_slice"]

    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="FBP2d_astra",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )
    assert res_norm < 1e-6


# ########################################################################
@pytest.mark.full_data
def test_pipe_FBP3d_tomobar_denoising_i13_177906_preview(
    get_files: Callable,
    cmd,
    i13_177906,
    FBP3d_tomobar_denoising,
    FBP3d_tomobar_TVdenoising_i13_177906_npz,
    output_folder,
):

    change_value_parameters_method_pipeline(
        FBP3d_tomobar_denoising,
        method=[
            "standard_tomo",
            "remove_all_stripe",
        ],
        key=[
            "preview",
            "normalize",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
            True,
        ],
    )

    # do not save the result of FBP3d_tomobar
    change_value_parameters_method_pipeline(
        FBP3d_tomobar_denoising,
        method=[
            "FBP3d_tomobar",
        ],
        key=[
            "recon_size",
        ],
        value=[
            None,
        ],
        save_result=False,
    )

    # change detector_pad value
    change_value_parameters_method_pipeline(
        FBP3d_tomobar_denoising,
        method=[
            "FBP3d_tomobar",
        ],
        key=[
            "detector_pad",
        ],
        value=[
            100,
        ],
    )

    # save the result of denoising instead
    change_value_parameters_method_pipeline(
        FBP3d_tomobar_denoising,
        method=[
            "total_variation_PD",
            "total_variation_PD",
        ],
        key=[
            "regularisation_parameter",
            "iterations",
        ],
        value=[
            1.0e-04,
            25,
        ],
        save_result=True,
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, i13_177906)
    cmd.insert(7, FBP3d_tomobar_denoising)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP3d_tomobar_TVdenoising_i13_177906_npz["data"]
    axis_slice = FBP3d_tomobar_TVdenoising_i13_177906_npz["axis_slice"]
    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="total_variation_PD",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )

    assert res_norm < 0.1


# ########################################################################


@pytest.mark.full_data
def test_pipe_360deg_paganin_FBP3d_tomobar_i13_179623_preview(
    get_files: Callable,
    cmd,
    i13_179623,
    deg360_paganin_FBP3d_tomobar,
    FBP3d_tomobar_paganin_i13_179623_npz,
    output_folder,
):
    change_value_parameters_method_pipeline(
        deg360_paganin_FBP3d_tomobar,
        method=[
            "standard_tomo",
            "find_center_360",
            "paganin_filter",
            "paganin_filter",
        ],
        key=[
            "preview",
            "ind",
            "energy",
            "ratio_delta_beta",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
            "mid",
            15.0,
            200,
        ],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, i13_179623)
    cmd.insert(7, deg360_paganin_FBP3d_tomobar)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP3d_tomobar_paganin_i13_179623_npz["data"]
    axis_slice = FBP3d_tomobar_paganin_i13_179623_npz["axis_slice"]
    res_norm = calculate_gt_residual(
        path_to_data="data/",
        h5_file_name="FBP3d_tomobar",
        h5_files=h5_files,
        data_gt=data_gt,
        axis_slice=axis_slice,
    )

    assert res_norm < 1e-4


# ########################################################################
@pytest.mark.full_data
def test_pipe_sweep_FBP3d_tomobar_i13_177906(
    get_files: Callable,
    cmd,
    i13_177906,
    sweep_center_FBP3d_tomobar,
    pipeline_sweep_FBP3d_tomobar_i13_177906_tiffs,
    output_folder,
):
    cmd.pop(4)  #: don't save all
    cmd.insert(5, i13_177906)
    cmd.insert(7, sweep_center_FBP3d_tomobar)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files_references = get_files(pipeline_sweep_FBP3d_tomobar_i13_177906_tiffs)

    # recurse through output_dir and check that all files are there
    files = get_files(output_folder)
    assert len(files) == 11

    #: check the number of the resulting tif files
    check_tif(files, 8, (2560, 2560))
    compare_tif(files, files_references)


# ########################################################################
@pytest.mark.full_data
def test_pipe_sweep_paganin_FBP3d_tomobar_i12_119647(
    get_files: Callable,
    cmd,
    i12_119647,
    sweep_paganin_FBP3d_tomobar,
    pipeline_paganin_sweep_paganin_images_i12_119647_tiffs,
    pipeline_paganin_sweep_recon_images_i12_119647_tiffs,
    output_folder,
):

    cmd.pop(4)  #: don't save all
    cmd.insert(5, i12_119647)
    cmd.insert(7, sweep_paganin_FBP3d_tomobar)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files_references_paganin = get_files(
        pipeline_paganin_sweep_paganin_images_i12_119647_tiffs
    )
    files_references_recon = get_files(
        pipeline_paganin_sweep_recon_images_i12_119647_tiffs
    )

    # recurse through output_dir and check that all files are there
    path_to_files_paganin = os.path.join(
        output_folder,
        os.listdir(output_folder)[0],
        "images_sweep_paganin_filter32bit_tif",
    )
    path_to_files_recon = os.path.join(
        output_folder,
        os.listdir(output_folder)[0],
        "images_sweep_FBP3d_tomobar32bit_tif",
    )

    files_paganin = get_files(path_to_files_paganin)
    assert len(files_paganin) == 3

    #: check the number of the resulting tif files
    check_tif(files_paganin, 3, (1801, 2560))
    # compare_tif(files_paganin, files_references_paganin)

    files_recon = get_files(path_to_files_recon)
    assert len(files_recon) == 3

    #: check the number of the resulting tif files
    check_tif(files_recon, 3, (2560, 2560))
    # compare_tif(files_recon, files_references_recon)


# ########################################################################
