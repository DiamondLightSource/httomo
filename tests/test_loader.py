import pytest
import subprocess
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD

from httomo.data.hdf.loaders import standard_tomo


@pytest.mark.cupy
def test_tomo_standard_testing_pipeline_loaded(
    cmd, standard_data, standard_loader, output_folder, testing_pipeline, merge_yamls
):
    cmd.pop(3)  #: don't save all
    cmd.insert(5, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(6, "temp.yaml")

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


@pytest.mark.cupy
def test_diad_testing_pipeline_loaded(
    cmd, diad_data, diad_loader, output_folder, testing_pipeline, merge_yamls
):
    cmd.insert(6, diad_data)
    merge_yamls(diad_loader, testing_pipeline)
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

    assert len(output) == 7

    # data
    assert output[0].sum() == 141348397
    np.testing.assert_allclose(output[0].mean(), 981.58574794, rtol=1e-5)
    assert output[0].shape == (180, 5, 160)

    # flats
    assert output[1].sum() == 15625259
    np.testing.assert_allclose(output[1].mean(), 976.5786875, rtol=1e-5)
    assert output[1].shape == (20, 5, 160)

    # angles
    assert output[3].shape == (180,)
    np.testing.assert_allclose(output[3].sum(), 281.1725424962865, rtol=1e-5)
    np.testing.assert_allclose(output[3].mean(), 1.562069680534925, rtol=1e-5)

    assert output[4] == 180  # angles_total
    assert output[5] == 5  # detector_y
    assert output[6] == 160  # detector_x
