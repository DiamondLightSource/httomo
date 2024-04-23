from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from shutil import copy
from typing import Optional, Union

import click
import logging
from mpi4py import MPI

import httomo.globals
from httomo.logger import setup_logger
from httomo.yaml_checker import validate_yaml_config
from httomo.runner.task_runner import TaskRunner
from httomo.ui_layer import UiLayer

from . import __version__

log = logging.getLogger(__name__)

@click.group
@click.version_option(version=__version__, message="%(version)s")
def main():
    """httomo: Software for High Throughput Tomography in parallel beam.
     
      Use `python -m httomo run --help` for more help on the runner. 
    """
    pass

@main.command()
@click.argument("in_data_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "yaml_config", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "out_dir",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--save-all",
    is_flag=True,
    help="Save intermediate datasets for all tasks in the pipeline.",
)
@click.option(
    "--gpu-id",
    type=click.INT,
    default=-1,
    help="The GPU ID of the device to use.",
)
@click.option(
    "--reslice-dir",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    default=None,
    help="Enable reslicing through the disk by providing a folder path to store intermediate file)",
)
def run(
    in_data_file: Path,
    yaml_config: Path,
    out_dir: Path,
    gpu_id: int,
    save_all: bool,
    reslice_dir: Union[Path, None],
):
    # Define httomo.globals.run_out_dir in all MPI processes
    httomo.globals.run_out_dir = out_dir.joinpath(
            f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # Setup global logger object
        httomo.globals.logger = setup_logger(httomo.globals.run_out_dir)

        # Copy YAML pipeline file to output directory
        copy(yaml_config, httomo.globals.run_out_dir)

    # try to access the GPU with the ID given
    try:
        import cupy as cp

        gpu_count = cp.cuda.runtime.getDeviceCount()

        if gpu_id != -1:
            if gpu_id not in range(0, gpu_count):
                raise ValueError(
                    f"GPU Device not available for access. Use a GPU ID in the range: 0 to {gpu_count} (exclusive)"
                )

            cp.cuda.Device(gpu_id).use()

        httomo.globals.gpu_id = gpu_id

    except ImportError:
        pass  # silently pass and run if the CPU pipeline is given

    # instantiate UiLayer class for pipeline build
    init_UiLayer = UiLayer(yaml_config, in_data_file, comm=comm)
    pipeline = init_UiLayer.build_pipeline()
    
    # Run the pipeline using Taskrunner
    runner = TaskRunner(pipeline, save_all, reslice_dir)
    return runner.execute()

def _check_yaml(yaml_config: Path, in_data: Path):
    """Check a YAML pipeline file for errors."""    
    return validate_yaml_config(yaml_config, in_data)