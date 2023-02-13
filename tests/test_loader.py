import os
import subprocess
import sys

if not os.path.exists("output_dir"):
    os.mkdir("output_dir/")


def test_tomo_standard_loaded():
    cmd = [
        sys.executable,
        "-m", "httomo",
        "testdata/tomo_standard.nxs",
        "samples/pipeline_template_examples/testing_pipeline.yaml",
        "output_dir/",
        "task_runner",
    ]
    output = subprocess.check_output(cmd).decode().strip()
    assert "Running task 1 (pattern=projection): standard_tomo" in output
    assert "Running task 2 (pattern=projection): normalize" in output
    assert "Running task 3 (pattern=projection): minus_log" in output
    assert "Running task 5 (pattern=sinogram): find_center_vo" in output
    assert "Running task 6 (pattern=sinogram): recon" in output
    assert "Running task 7 (pattern=all): save_to_images" in output
    assert "Total number of reslices: 1" in output
