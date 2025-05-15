from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from click.testing import CliRunner

import httomo
from httomo import __version__
from httomo.cli import set_global_constants, transform_limit_str_to_bytes, main


def test_cli_version_shows_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert __version__ == result.stdout.strip()


def test_cli_help_shows_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.stdout.strip().startswith("Usage: ")


def test_cli_noargs_shows_help():
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 2
    assert result.stderr.strip().startswith("Usage: ")


def test_cli_check_pass_data_file(standard_loader, standard_data):
    runner = CliRunner()
    result = runner.invoke(main, ["check", standard_loader, standard_data])
    check_data_str = (
        "Checking that the paths to the data and keys in the YAML_CONFIG file "
        "match the paths and keys in the input file (IN_DATA)..."
    )
    assert check_data_str in result.stdout


@pytest.mark.cupy
def test_cli_pass_gpu_id(standard_data, standard_loader, output_folder):
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", standard_data, standard_loader, output_folder, "--gpu-id", "100"]
    )
    assert "GPU Device not available for access." in str(result.exception)


@pytest.mark.parametrize(
    "cli_parameter,limit_bytes",
    [
        ("0", 0),
        ("500", 500),
        ("500k", 500 * 1024),
        ("1M", 1024 * 1024),
        ("1m", 1024 * 1024),
        ("3g", 3 * 1024**3),
        ("3.2g", int(3.2 * 1024**3)),
    ],
)
def test_cli_transforms_memory_limits(cli_parameter: str, limit_bytes: int):
    assert transform_limit_str_to_bytes(cli_parameter) == limit_bytes


@pytest.mark.parametrize("cli_parameter", ["abcd", "nolimit", "124A", "23ki"])
def test_cli_fails_transforming_memory_limits(cli_parameter: str):
    with pytest.raises(ValueError) as e:
        transform_limit_str_to_bytes(cli_parameter)

    assert f"invalid memory limit string {cli_parameter}" in str(e)


def test_output_folder_name_correctly_sets_run_out_dir_global_constant(output_folder):
    output_dir = "output_dir"  # dir created by the `output_folder` fixture
    dir_name = "test-output"  # subdir that should be created by httomo
    custom_output_dir = Path(output_dir, dir_name)
    set_global_constants(
        out_dir=Path(output_dir),
        intermediate_format="hdf5",
        compress_intermediate=False,
        frames_per_chunk=0,
        max_cpu_slices=1,
        syslog_host="localhost",
        syslog_port=514,
        output_folder_name=Path(dir_name),
        recon_filename_stem=None,
    )
    assert httomo.globals.run_out_dir == custom_output_dir


@pytest.mark.parametrize(
    "use_recon_filename_stem_flag",
    [True, False],
)
def test_cli_recon_filename_stem_flag(
    standard_data, standard_loader, output_folder, use_recon_filename_stem_flag: bool
):
    runner = CliRunner()
    if use_recon_filename_stem_flag:
        filename_stem = "my-file"
        runner.invoke(
            main,
            [
                "run",
                standard_data,
                standard_loader,
                output_folder,
                "--recon-filename-stem",
                filename_stem,
            ],
        )
        assert httomo.globals.RECON_FILENAME_STEM is not None
        assert httomo.globals.RECON_FILENAME_STEM == filename_stem
    else:
        runner.invoke(
            main,
            ["run", standard_data, standard_loader, output_folder],
        )
        assert httomo.globals.RECON_FILENAME_STEM is None
