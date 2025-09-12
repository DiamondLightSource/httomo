import subprocess
from typing import Callable, List, Tuple, Union

from subprocess import Popen, PIPE
import os


import h5py
import numpy as np
import pytest
from plumbum import local
from .conftest import change_value_parameters_method_pipeline, check_tif, compare_tif

# NOTE: those tests have path integrated that are compatible with running jobs in Jenkins at DLS infrastructure.

########################################################################


@pytest.mark.full_data_parallel
def test_pipe_parallel_FBP3d_tomobar_k11_38730_in_disk_preview(
    get_files: Callable,
    cmd_mpirun,
    diad_k11_38730,
    FBP3d_tomobar_noimagesaving,
    FBP3d_tomobar_k11_38730_npz,
    output_folder,
):

    change_value_parameters_method_pipeline(
        FBP3d_tomobar_noimagesaving,
        method=[
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
        ],
        key=[
            "data_path",
            "image_key_path",
            "rotation_angles",
            "preview",
        ],
        value=[
            "/entry/imaging/data",
            "/entry/instrument/imaging/image_key",
            {"data_path": "/entry/imaging_sum/gts_cs_theta"},
            {"detector_y": {"start": 500, "stop": 1500}},
        ],
    )

    # NOTE that the intermediate file with file-based processing will be saved to /scratch/jenkins_agent/workspace/
    cmd_mpirun.insert(9, diad_k11_38730)
    cmd_mpirun.insert(10, FBP3d_tomobar_noimagesaving)
    cmd_mpirun.insert(11, output_folder)
    cmd_mpirun.insert(12, "--max-memory")
    cmd_mpirun.insert(13, "5G")
    cmd_mpirun.insert(14, "--reslice-dir")
    cmd_mpirun.insert(15, "/scratch/jenkins_agent/workspace/")

    process = Popen(
        cmd_mpirun, env=os.environ, shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    output, error = process.communicate()
    print(output)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP3d_tomobar_k11_38730_npz["data"]
    axis_slice = FBP3d_tomobar_k11_38730_npz["axis_slice"]
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


@pytest.mark.full_data_parallel
def test_pipe_parallel_FBP3d_tomobar_k11_38730_in_memory_preview(
    get_files: Callable,
    cmd_mpirun,
    diad_k11_38730,
    FBP3d_tomobar_noimagesaving,
    FBP3d_tomobar_k11_38730_npz,
    output_folder,
):

    change_value_parameters_method_pipeline(
        FBP3d_tomobar_noimagesaving,
        method=[
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
        ],
        key=[
            "data_path",
            "image_key_path",
            "rotation_angles",
            "preview",
        ],
        value=[
            "/entry/imaging/data",
            "/entry/instrument/imaging/image_key",
            {"data_path": "/entry/imaging_sum/gts_cs_theta"},
            {"detector_y": {"start": 500, "stop": 1500}},
        ],
    )

    cmd_mpirun.insert(9, diad_k11_38730)
    cmd_mpirun.insert(10, FBP3d_tomobar_noimagesaving)
    cmd_mpirun.insert(11, output_folder)

    process = Popen(
        cmd_mpirun, env=os.environ, shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    output, error = process.communicate()
    print(output)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP3d_tomobar_k11_38730_npz["data"]
    axis_slice = FBP3d_tomobar_k11_38730_npz["axis_slice"]
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


@pytest.mark.full_data_parallel
def test_parallel_pipe_FBP3d_tomobar_denoising_i13_177906_preview(
    get_files: Callable,
    cmd_mpirun,
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

    cmd_mpirun.insert(9, i13_177906)
    cmd_mpirun.insert(10, FBP3d_tomobar_denoising)
    cmd_mpirun.insert(11, output_folder)

    process = Popen(
        cmd_mpirun, env=os.environ, shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    output, error = process.communicate()
    print(output)

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
    assert res_norm_tv_res < 1e-1


# ########################################################################


@pytest.mark.full_data_parallel
def test_parallel_pipe_LPRec3d_tomobar_i12_119647_preview(
    get_files: Callable,
    cmd_mpirun,
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

    cmd_mpirun.insert(9, i12_119647)
    cmd_mpirun.insert(10, LPRec3d_tomobar)
    cmd_mpirun.insert(11, output_folder)

    process = Popen(
        cmd_mpirun, env=os.environ, shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    output, error = process.communicate()
    print(output)

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


@pytest.mark.full_data_parallel
def test_parallel_pipe_360deg_distortion_FBP3d_tomobar_i13_179623_preview(
    get_files: Callable,
    cmd_mpirun,
    i13_179623,
    deg360_distortion_FBP3d_tomobar,
    FBP3d_tomobar_distortion_i13_179623_npz,
    output_folder,
):
    change_value_parameters_method_pipeline(
        deg360_distortion_FBP3d_tomobar,
        method=[
            "standard_tomo",
            "normalize",
            "find_center_360",
            "find_center_360",
            "find_center_360",
            "distortion_correction_proj_discorpy",
            "FBP3d_tomobar",
        ],
        key=[
            "preview",
            "minus_log",
            "ind",
            "use_overlap",
            "norm",
            "metadata_path",
            "neglog",
        ],
        value=[
            {"detector_y": {"start": 900, "stop": 1100}},
            False,
            24,
            True,
            True,
            "/data/tomography/raw_data/i13/360/179623_coeff.txt",
            True,
        ],
    )

    cmd_mpirun.insert(9, i13_179623)
    cmd_mpirun.insert(10, deg360_distortion_FBP3d_tomobar)
    cmd_mpirun.insert(11, output_folder)

    process = Popen(
        cmd_mpirun, env=os.environ, shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    output, error = process.communicate()
    print(output)

    files = get_files(output_folder)

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = FBP3d_tomobar_distortion_i13_179623_npz["data"]
    axis_slice = FBP3d_tomobar_distortion_i13_179623_npz["axis_slice"]
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
@pytest.mark.full_data_parallel
def test_parallel_pipe_sweep_FBP3d_tomobar_i13_177906(
    get_files: Callable,
    cmd_mpirun,
    i13_177906,
    sweep_center_FBP3d_tomobar,
    pipeline_sweep_FBP3d_tomobar_i13_177906_tiffs,
    output_folder,
):

    cmd_mpirun.insert(9, i13_177906)
    cmd_mpirun.insert(10, sweep_center_FBP3d_tomobar)
    cmd_mpirun.insert(11, output_folder)

    process = Popen(
        cmd_mpirun, env=os.environ, shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    output, error = process.communicate()
    print(output)

    files = get_files(output_folder)
    files_references = get_files(pipeline_sweep_FBP3d_tomobar_i13_177906_tiffs)

    # recurse through output_dir and check that all files are there
    files = get_files(output_folder)
    assert len(files) == 11

    #: check the number of the resulting tif files
    check_tif(files, 8, (2560, 2560))
    compare_tif(files, files_references)


# ########################################################################
