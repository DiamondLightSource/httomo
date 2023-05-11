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
