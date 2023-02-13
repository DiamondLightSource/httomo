# Defines common fixtures and makes them available to all tests

import pytest
from pathlib import Path
from shutil import rmtree


@pytest.fixture
def clean_folder():
    for path in Path("output_dir").iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
