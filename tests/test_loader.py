import subprocess

import numpy as np
import pytest
from mpi4py import MPI

from httomo.data.hdf.loaders import standard_tomo

comm = MPI.COMM_WORLD



def test_tomo_standard_testing_pipeline_loaded(
    cmd, standard_data, standard_loader, output_folder, testing_pipeline, merge_yamls
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(7, "temp.yaml")

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    assert "Running task 1 (pattern=projection): standard_tomo..." in result.stderr
    assert "Running task 2 (pattern=projection): normalize..." in result.stderr
    assert "Running task 3 (pattern=projection): minus_log.." in result.stderr
    assert "Running task 4 (pattern=sinogram): remove_stripe_fw..." in result.stderr
    assert "Running task 5 (pattern=sinogram): find_center_vo..." in result.stderr
    assert "Running task 7 (pattern=all): save_to_images.." in result.stderr
    assert "Pipeline finished" in result.stderr


def test_diad_testing_pipeline_loaded(
    cmd, diad_data, diad_loader, output_folder, testing_pipeline, merge_yamls
):
    cmd.insert(7, diad_data)
    merge_yamls(diad_loader, testing_pipeline)
    cmd.insert(8, "temp.yaml")

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert "Running task 1 (pattern=projection): standard_tomo..." in result.stderr
    assert "Running task 2 (pattern=projection): normalize..." in result.stderr
    assert "Running task 3 (pattern=projection): minus_log.." in result.stderr
    assert "Running task 4 (pattern=sinogram): remove_stripe_fw..." in result.stderr
    assert "Running task 5 (pattern=sinogram): find_center_vo..." in result.stderr
    assert "Running task 7 (pattern=all): save_to_images.." in result.stderr
    assert "Pipeline finished" in result.stderr


def test_standard_tomo(standard_data, standard_data_path, standard_image_key_path):
    preview = [None, {"start": 5, "stop": 10}, None]

    output = standard_tomo(
        "tomo",
        standard_data,
        standard_data_path,
        1,
        preview,
        1,
        comm,
        image_key_path=standard_image_key_path,
    )

    # data
    assert output.data.sum() == 141348397
    np.testing.assert_allclose(output.data.mean(), 981.58574794, rtol=1e-5)
    assert output.data.shape == (180, 5, 160)

    # flats
    assert output.flats.sum() == 15625259
    np.testing.assert_allclose(output.flats.mean(), 976.5786875, rtol=1e-5)
    assert output.flats.shape == (20, 5, 160)

    # angles
    assert output.angles.shape == (180,)
    np.testing.assert_allclose(output.angles.sum(), 281.1725424962865, rtol=1e-5)
    np.testing.assert_allclose(output.angles.mean(), 1.562069680534925, rtol=1e-5)

    assert output.angles_total == 180  # angles_total
    assert output.detector_y == 5  # detector_y
    assert output.detector_x == 160  # detector_x


def test_standard_tomo_ignore_darks(
    standard_data, standard_data_path, standard_image_key_path
):
    preview = [None, None, None]
    # Ignore 3 individual darks
    ignore_darks = {"individual": [215, 217, 219]}
    output = standard_tomo(
        "tomo",
        standard_data,
        standard_data_path,
        1,
        preview,
        1,
        comm,
        image_key_path=standard_image_key_path,
        ignore_darks=ignore_darks,
    )
    assert output.darks.shape[0] == 17  # 3 darks removed
    assert output.flats.shape[0] == 20  # no flats removed


def test_standard_tomo_ignore_flats(
    standard_data, standard_data_path, standard_image_key_path
):
    preview = [None, None, None]
    # Ignore 5 batch flats
    ignore_flats = {
        "batch": [
            {
                "start": 185,
                "stop": 189,
            }
        ]
    }
    output = standard_tomo(
        "tomo",
        standard_data,
        standard_data_path,
        1,
        preview,
        1,
        comm,
        image_key_path=standard_image_key_path,
        ignore_flats=ignore_flats,
    )
    assert output.darks.shape[0] == 20  # no darks removed
    assert output.flats.shape[0] == 15  # 5 flats removed


def test_standard_tomo_ignore_flats_error(
    standard_data, standard_data_path, standard_image_key_path
):
    preview = [None, None, None]
    # Ignore 5 batch flats, one of which (the end one) is outside the range of
    # flats
    ignore_flats = {
        "batch": [
            {
                "start": 195,
                "stop": 200,
            }
        ]
    }
    expected_err_str = (
        r"The flats indices to ignore are \[195, 196, 197, 198, 199, 200\], "
        r"which has one or more values outside the flats in the dataset."
    )
    with pytest.raises(ValueError, match=expected_err_str):
        output = standard_tomo(
            "tomo",
            standard_data,
            standard_data_path,
            1,
            preview,
            1,
            comm,
            image_key_path=standard_image_key_path,
            ignore_flats=ignore_flats,
        )


def test_diad_loader():
    in_file = "tests/test_data/k11_diad/k11-18014.nxs"
    data_path = "/entry/imaging/data"
    image_key_path = "/entry/instrument/imaging/image_key"
    rotation_angles = {"data_path": "/entry/imaging_sum/gts_theta_value"}

    output = standard_tomo(
        "tomo",
        in_file,
        data_path,
        1,
        [None, {"start": 5, "stop": 7}, None],
        0,
        comm,
        image_key_path=image_key_path,
        rotation_angles=rotation_angles,
    )

    assert output.data.sum() == 6019533062
    np.testing.assert_allclose(output.data.mean(), 38573.89243329147, rtol=1e-5)
    assert output.data.shape == (3001, 2, 26)

    assert output.flats.sum() == 236484277
    np.testing.assert_allclose(output.flats.mean(), 45477.74557692308, rtol=1e-5)
    assert output.flats.shape == (100, 2, 26)

    assert output.angles.shape == (3001,)
    np.testing.assert_allclose(output.angles.sum(), 9427.76925660484, rtol=1e-5)
    np.testing.assert_allclose(output.angles.mean(), 3.1415425713444987, rtol=1e-5)

    assert output.angles_total == 3001
    assert output.detector_y == 2
    assert output.detector_x == 26
