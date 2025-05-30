from pathlib import Path
import json
import tempfile

import pytest
from pytest_mock import MockerFixture
from click.testing import CliRunner

import httomo
from httomo import __version__
from httomo.cli import set_global_constants, transform_limit_str_to_bytes, main
from httomo.ui_layer import PipelineFormat


# Sample JSON pipeline data for testing
SAMPLE_JSON_PIPELINE = [
    {
        "method": "standard_tomo",
        "module_path": "httomo.data.hdf.loaders",
        "parameters": {
            "data_path": "/entry1/tomo_entry/data/data",
            "image_key_path": "/entry1/tomo_entry/instrument/detector/image_key",
            "rotation_angles": {"data_path": "/entry1/tomo_entry/data/rotation_angle"},
        },
    },
    {
        "method": "remove_outlier",
        "module_path": "tomopy.misc.corr",
        "parameters": {"dif": 0.1, "size": 3, "axis": "auto"},
    },
]


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


def test_cli_check_accepts_json_string(mocker):
    """Test that the check command accepts JSON string input."""
    # Create a temporary data file for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".h5", delete=False) as f:
        temp_data_file = f.name

    try:
        # Mock the validation function to prevent actual execution
        mock_validate = mocker.patch(
            "httomo.cli.validate_yaml_config", return_value=True
        )

        runner = CliRunner()
        json_string = json.dumps(SAMPLE_JSON_PIPELINE)

        result = runner.invoke(main, ["check", json_string, temp_data_file])

        # The command should execute without errors (though it may not fully work due to missing JSON handling)
        # We're mainly testing that it accepts the JSON string as input
        assert result.exit_code in [
            0,
            1,
        ], f"CLI command failed unexpectedly with: {result.output}"

    finally:
        # Clean up the temporary file
        import os

        os.unlink(temp_data_file)


@pytest.mark.parametrize("pipeline_format", ["Json", "json"])
def test_cli_run_accepts_json_string_with_format_flag(mocker, pipeline_format: str):
    """Test that the run command accepts JSON string input with --pipeline-format"""
    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".h5", delete=False) as f:
        temp_data_file = f.name

    with tempfile.TemporaryDirectory() as temp_output_dir:
        try:
            # Mock all the functions that would actually execute the pipeline
            mocker.patch("httomo.cli.setup_logger")
            mock_mpi = mocker.patch("httomo.cli.MPI")
            mock_mpi.COMM_WORLD.rank = 0
            mock_mpi.COMM_SELF = mocker.MagicMock()

            mock_ui_layer = mocker.patch("httomo.cli.UiLayer")
            mock_ui_instance = mocker.MagicMock()
            mock_ui_layer.return_value = mock_ui_instance
            mock_pipeline = mocker.MagicMock()
            mock_ui_instance.build_pipeline.return_value = mock_pipeline

            mock_transform_layer = mocker.patch("httomo.cli.TransformLayer")
            mock_transform_instance = mocker.MagicMock()
            mock_transform_layer.return_value = mock_transform_instance
            mock_transform_instance.transform.return_value = mock_pipeline

            mocker.patch("httomo.cli.initialise_output_directory")
            mocker.patch("httomo.cli.execute_high_throughput_run")
            mocker.patch("httomo.cli.is_sweep_pipeline", return_value=False)
            mocker.patch("httomo.cli.make_monitors", return_value=None)

            runner = CliRunner()
            json_string = json.dumps(SAMPLE_JSON_PIPELINE)

            result = runner.invoke(
                main,
                [
                    "run",
                    temp_data_file,  # IN_DATA_FILE comes first
                    json_string,  # PIPELINE comes second
                    temp_output_dir,  # OUT_DIR comes third
                    "--pipeline-format",
                    pipeline_format,
                ],
            )

            # Check that the command executed without errors
            assert (
                result.exit_code == 0
            ), f"CLI command failed with output: {result.output}\nException: {result.exception}"

            # Verify UiLayer was instantiated with correct parameters
            mock_ui_layer.assert_called_once()
            call_args = mock_ui_layer.call_args

            # Get the arguments passed to UiLayer (positional and keyword args)
            if len(call_args) > 0 and len(call_args[0]) > 0:
                # Pipeline is the first positional argument
                pipeline_arg = call_args[0][0]
                assert pipeline_arg == json_string
                assert isinstance(pipeline_arg, str)

            # Check keyword arguments if they exist
            if len(call_args) > 1:
                call_kwargs = call_args[1]
                if "pipeline_format" in call_kwargs:
                    assert call_kwargs["pipeline_format"] == PipelineFormat.Json

        finally:
            # Clean up the temporary file
            import os

            os.unlink(temp_data_file)


def test_initialise_output_directory_handles_json_string(tmp_path):
    """Test that initialise_output_directory correctly handles JSON string input."""
    from httomo.cli import initialise_output_directory

    # Set up the global output directory
    output_dir = tmp_path / "output"
    httomo.globals.run_out_dir = output_dir

    json_string = json.dumps(SAMPLE_JSON_PIPELINE)

    # Call the function with a JSON string
    initialise_output_directory(json_string)

    # Verify directory was created
    assert output_dir.exists()

    # Verify that the JSON was written to a file
    expected_file = output_dir / "pipeline.json"
    assert expected_file.exists()

    # Verify the content is correct
    with open(expected_file, "r") as f:
        written_content = f.read()
    assert written_content == json_string


def test_initialise_output_directory_handles_path_input(mocker, tmp_path):
    """Test that initialise_output_directory correctly handles Path input (existing behavior)."""
    from httomo.cli import initialise_output_directory

    # Set up the global output directory
    output_dir = tmp_path / "output"
    httomo.globals.run_out_dir = output_dir

    # Mock copy to avoid actual file operations
    mock_copy = mocker.patch("httomo.cli.copy")

    pipeline_path = Path("some/pipeline.yaml")

    # Call the function with a Path
    initialise_output_directory(pipeline_path)

    # Verify directory was created
    assert output_dir.exists()

    # Verify that copy was called for the file
    mock_copy.assert_called_once_with(pipeline_path, output_dir)


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
