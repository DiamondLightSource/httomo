import subprocess
from typing import Callable, List, Tuple, Union
import h5py
import numpy as np
import pytest
import os
from plumbum import local
from .conftest import change_value_parameters_method_pipeline, check_tif, compare_tif

# NOTE: those tests have path integrated that are compatible with running jobs in Jenkins at DLS infrastructure.


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
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "FBP3d_tomobar"
    for file_to_open in h5_files:
        if h5_file_name in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            index_prog = step
            for i in range(slices):
                data_result[i, :, :] = h5f[path_to_data][:, index_prog, :]
                index_prog += step
            h5f.close()
        else:
            message_str = f"File name with {h5_file_name} string cannot be found."
            raise FileNotFoundError(message_str)

    residual_im = data_gt - data_result
    res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
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
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "FBP3d_tomobar"
    for file_to_open in h5_files:
        if h5_file_name in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            index_prog = step
            for i in range(slices):
                data_result[i, :, :] = h5f[path_to_data][:, index_prog, :]
                index_prog += step
            h5f.close()
        else:
            message_str = f"File name with {h5_file_name} string cannot be found."
            raise FileNotFoundError(message_str)

    residual_im = data_gt - data_result
    res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
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
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "LPRec3d_tomobar"
    for file_to_open in h5_files:
        if h5_file_name in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            index_prog = step
            for i in range(slices):
                data_result[i, :, :] = h5f[path_to_data][:, index_prog, :]
                index_prog += step
            h5f.close()
        else:
            message_str = f"File name with {h5_file_name} string cannot be found."
            raise FileNotFoundError(message_str)

    residual_im = data_gt - data_result
    res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
    assert (
        res_norm < 0.2
    )  # TODO: known issue with the Log-Polar, the tolerance will be reduced when fixed


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
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "FBP2d_astra"
    for file_to_open in h5_files:
        if h5_file_name in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            index_prog = step
            for i in range(slices):
                data_result[i, :, :] = h5f[path_to_data][:, index_prog, :]
                index_prog += step
            h5f.close()
        else:
            message_str = f"File name with {h5_file_name} string cannot be found."
            raise FileNotFoundError(message_str)

    residual_im = data_gt - data_result
    res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
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
        ],
        key=[
            "preview",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
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
    data_gt_tv = FBP3d_tomobar_TVdenoising_i13_177906_npz["data"]
    axis_slice = FBP3d_tomobar_TVdenoising_i13_177906_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt_tv)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "total_variation_PD"
    for file_to_open in h5_files:
        if h5_file_name in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            index_prog = step
            for i in range(slices):
                data_result[i, :, :] = h5f[path_to_data][:, index_prog, :]
                index_prog += step
            h5f.close()
        else:
            message_str = f"File name with {h5_file_name} string cannot be found."
            raise FileNotFoundError(message_str)

    residual_im = data_gt_tv - data_result
    res_norm_tv_res = np.linalg.norm(residual_im.flatten()).astype("float32")
    assert res_norm_tv_res < 1e-5


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
            "normalize",
            "find_center_360",
            "paganin_filter_tomopy",
            "paganin_filter_tomopy",
        ],
        key=[
            "preview",
            "minus_log",
            "ind",
            "energy",
            "alpha",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
            False,
            "mid",
            15.0,
            0.1,
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
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "FBP3d_tomobar"
    for file_to_open in h5_files:
        if h5_file_name in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            index_prog = step
            for i in range(slices):
                data_result[i, :, :] = h5f[path_to_data][:, index_prog, :]
                index_prog += step
            h5f.close()
        else:
            message_str = f"File name with {h5_file_name} string cannot be found."
            raise FileNotFoundError(message_str)

    residual_im = data_gt - data_result
    res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
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

    files_references_paganin = get_files(pipeline_paganin_sweep_paganin_images_i12_119647_tiffs)
    files_references_recon= get_files(pipeline_paganin_sweep_recon_images_i12_119647_tiffs)

    # recurse through output_dir and check that all files are there
    path_to_files_paganin =  os.path.join(output_folder, os.listdir(output_folder)[0], "images_sweep_paganin_filter_tomopy8bit_tif")
    path_to_files_recon =  os.path.join(output_folder, os.listdir(output_folder)[0], "images_sweep_FBP3d_tomobar16bit_tif")
    
    files_paganin = get_files(path_to_files_paganin)
    assert len(files_paganin) == 3

    #: check the number of the resulting tif files
    check_tif(files_paganin, 3, (1801, 2560))
    compare_tif(files_paganin, files_references_paganin)

    files_recon = get_files(path_to_files_recon)
    assert len(files_recon) == 3

    #: check the number of the resulting tif files
    check_tif(files_recon, 3, (2560, 2560))
    compare_tif(files_recon, files_references_recon)
# ########################################################################
