import subprocess


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
