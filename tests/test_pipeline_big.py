import re
import subprocess
from typing import Callable, List, Tuple, Union

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from plumbum import local
from .conftest import change_value_parameters_method_pipeline

# NOTE: those tests have path integrated that are compatible with running jobs in Jenkins at DLS infrastructure.


@pytest.mark.full_data
def test_pipeline_gpu_FBP_diad_k11_38731_in_disk(
    get_files: Callable,
    cmd,
    diad_k11_38731,
    gpu_pipeline_diad_FBP_noimagesaving,
    gpu_diad_FBP_k11_38731_npz,
    output_folder,
):
    # NOTE that the intermediate file with file-based processing will be saved to /tmp

    cmd.pop(4)  #: don't save all
    cmd.insert(5, diad_k11_38731)
    cmd.insert(7, gpu_pipeline_diad_FBP_noimagesaving)
    cmd.insert(8, output_folder)
    cmd.insert(9, "--max-memory")
    cmd.insert(10, "40G")
    cmd.insert(11, "--reslice-dir")
    cmd.insert(12, "/scratch/jenkins_agent/workspace/")

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = gpu_diad_FBP_k11_38731_npz["data"]
    axis_slice = gpu_diad_FBP_k11_38731_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "httomolibgpu-FBP"
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


@pytest.mark.full_data
def test_pipeline_gpu_FBP_diad_k11_38731_in_memory(
    get_files: Callable,
    cmd,
    diad_k11_38731,
    gpu_pipeline_diad_FBP_noimagesaving,
    gpu_diad_FBP_k11_38731_npz,
    output_folder,
):
    cmd.pop(4)  #: don't save all
    cmd.insert(5, diad_k11_38731)
    cmd.insert(7, gpu_pipeline_diad_FBP_noimagesaving)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = gpu_diad_FBP_k11_38731_npz["data"]
    axis_slice = gpu_diad_FBP_k11_38731_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "httomolibgpu-FBP"
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


@pytest.mark.full_data
def test_pipeline_gpu_FBP_diad_k11_38730_in_disk(
    get_files: Callable,
    cmd,
    diad_k11_38730,
    gpu_pipeline_diad_FBP_noimagesaving,
    gpu_diad_FBP_k11_38730_npz,
    output_folder,
):
    # NOTE that the intermediate file with file-based processing will be saved to /tmp

    cmd.pop(4)  #: don't save all
    cmd.insert(5, diad_k11_38730)
    cmd.insert(7, gpu_pipeline_diad_FBP_noimagesaving)
    cmd.insert(8, output_folder)
    cmd.insert(9, "--max-memory")
    cmd.insert(10, "40G")
    cmd.insert(11, "--reslice-dir")
    cmd.insert(12, "/scratch/jenkins_agent/workspace/")

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = gpu_diad_FBP_k11_38730_npz["data"]
    axis_slice = gpu_diad_FBP_k11_38730_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "httomolibgpu-FBP"
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


@pytest.mark.full_data
def test_pipeline_gpu_FBP_diad_k11_38730_in_memory(
    get_files: Callable,
    cmd,
    diad_k11_38730,
    gpu_pipeline_diad_FBP_noimagesaving,
    gpu_diad_FBP_k11_38730_npz,
    output_folder,
):
    cmd.pop(4)  #: don't save all
    cmd.insert(5, diad_k11_38730)
    cmd.insert(7, gpu_pipeline_diad_FBP_noimagesaving)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = gpu_diad_FBP_k11_38730_npz["data"]
    axis_slice = gpu_diad_FBP_k11_38730_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "httomolibgpu-FBP"
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


@pytest.mark.full_data
def test_pipeline_gpu_FBP_denoising_i13_177906_preview(
    get_files: Callable,
    cmd,
    i13_177906,
    gpu_pipelineFBP_denoising,
    gpu_FBP_TVdenoising_i13_177906_npz,
    output_folder,
):

    change_value_parameters_method_pipeline(
        gpu_pipelineFBP_denoising,
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

    # do not save the result of FBP
    change_value_parameters_method_pipeline(
        gpu_pipelineFBP_denoising,
        method=[
            "FBP",
        ],
        key=[
            "recon_size",
        ],
        value=[
            None,
        ],
        save_result=False,
    )

    # save the result of denoising instead
    change_value_parameters_method_pipeline(
        gpu_pipelineFBP_denoising,
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
    cmd.insert(7, gpu_pipelineFBP_denoising)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = gpu_FBP_TVdenoising_i13_177906_npz["data"]
    axis_slice = gpu_FBP_TVdenoising_i13_177906_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt)

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

    residual_im = data_gt - data_result
    res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
    assert res_norm < 1e-6


@pytest.mark.full_data
def test_pipeline_gpu_360_paganin_FBP_i13_179623_preview(
    get_files: Callable,
    cmd,
    i13_179623,
    gpu_pipeline_360_paganin_FBP,
    gpu_FBP_paganin_i13_179623_npz,
    output_folder,
):
    change_value_parameters_method_pipeline(
        gpu_pipeline_360_paganin_FBP,
        method=[
            "standard_tomo",
            "normalize",
            "find_center_360",
            "sino_360_to_180",
            "paganin_filter_tomopy",
            "paganin_filter_tomopy",
        ],
        key=[
            "preview",
            "minus_log",
            "ind",
            "rotation",
            "energy",
            "alpha",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1200}},
            False,
            "mid",
            "right",
            15.0,
            0.1,
        ],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(5, i13_179623)
    cmd.insert(7, gpu_pipeline_360_paganin_FBP)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = gpu_FBP_paganin_i13_179623_npz["data"]
    axis_slice = gpu_FBP_paganin_i13_179623_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    h5_file_name = "httomolibgpu-FBP"
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
