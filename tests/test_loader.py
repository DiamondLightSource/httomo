import subprocess


def test_tomo_standard_loaded(
    cmd,
    standard_data,
    standard_loader,
):
    cmd.pop(3) #: don't save all
    cmd.insert(5, standard_data)
    cmd.insert(6, standard_loader)
    output = subprocess.check_output(cmd).decode().strip()

    assert "dataset shape is (220, 128, 160)" in output
    assert standard_data in output
    assert "Data shape is (180, 128, 160) of type uint16" in output


def test_tomo_standard_testing_pipeline_loaded(
    cmd,
    standard_data,
    testing_pipeline,
):
    cmd.pop(3) #: don't save all
    cmd.insert(5, standard_data)
    cmd.insert(6, testing_pipeline)
    output = subprocess.check_output(cmd).decode().strip()

    assert "Running task 1 (pattern=projection): standard_tomo" in output
    assert "Running task 2 (pattern=projection): normalize" in output
    assert "Running task 3 (pattern=projection): minus_log" in output
    assert "Running task 5 (pattern=sinogram): find_center_vo" in output
    assert "Running task 6 (pattern=sinogram): recon" in output
    assert "Running task 7 (pattern=all): save_to_images" in output
    assert "Total number of reslices: 1" in output


def test_tomo_standard_testing_pipeline_loaded_with_save_all(
    cmd,
    standard_data,
    testing_pipeline,
):
    cmd.insert(6, standard_data)
    cmd.insert(7, testing_pipeline)
    output = subprocess.check_output(cmd).decode().strip()

    assert "Saving intermediate file: 2-tomopy-normalize-tomo.h5" in output
    assert "Saving intermediate file: 3-tomopy-minus_log-tomo.h5" in output
    assert "Saving intermediate file: 4-tomopy-remove_stripe_fw-tomo.h5" in output
    assert "Saving intermediate file: 6-tomopy-recon-tomo-gridrec.h5" in output


def test_k11_diad_loaded(
    cmd,
    diad_data,
    diad_loader,
):
    cmd.insert(6, diad_data)
    cmd.insert(7, diad_loader)
    output = subprocess.check_output(cmd).decode().strip()

    assert "dataset shape is (3201, 22, 26)" in output
    assert diad_data in output
    assert "Data shape is (3001, 22, 26) of type uint16" in output


def test_diad_testing_pipeline_loaded(
    cmd,
    diad_data,
    diad_testing_pipeline,
):
    cmd.insert(6, diad_data)
    cmd.insert(7, diad_testing_pipeline)
    output = subprocess.check_output(cmd).decode().strip()

    assert "Running task 1 (pattern=projection): standard_tomo..." in output
    assert "Running task 3 (pattern=projection): minus_log..." in output
    assert "Saving intermediate file: 3-tomopy-minus_log-tomo.h5" in output
    assert "Running task 4 (pattern=sinogram): remove_all_stripe..." in output
    assert "Saving intermediate file: 4-tomopy-remove_all_stripe-tomo.h5" in output
    assert "Running task 6 (pattern=all): save_to_images..." in output
