import pytest
import subprocess
import sys

from httomo import __version__


@pytest.mark.cupy
def test_cli_version_shows_version():
    cmd = [sys.executable, "-m", "httomo", "--version"]
    assert __version__ == subprocess.check_output(cmd).decode().strip()


@pytest.mark.cupy
def test_cli_help_shows_help():
    cmd = [sys.executable, "-m", "httomo", "--help"]
    assert (
        subprocess.check_output(cmd)
        .decode()
        .strip()
        .startswith("Usage: python -m httomo")
    )


@pytest.mark.cupy
def test_cli_noargs_raises_error():
    cmd = [sys.executable, "-m", "httomo"]
    try:
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        assert e.returncode == 2


@pytest.mark.cupy
def test_cli_check_pass_data_file(standard_loader, standard_data):
    cmd = [sys.executable, "-m", "httomo", "check", standard_loader, standard_data]
    check_data_str = (
        "Checking that the paths to the data and keys in the YAML_CONFIG file "
        "match the paths and keys in the input file (IN_DATA)..."
    )
    assert check_data_str in subprocess.check_output(cmd).decode().strip()
