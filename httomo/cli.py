from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from shutil import copy
from typing import Optional

import click
from mpi4py import MPI

import httomo.globals
from httomo.common import PipelineTasks
from httomo.logger import setup_logger
from httomo.task_runner import run_tasks
from httomo.yaml_checker import validate_yaml_config

from . import __version__


@click.group
@click.version_option(version=__version__, message="%(version)s")
def main():
    """httomo: High Throughput Tomography."""
    pass


@main.command()
@click.argument(
    "yaml_config", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "in_data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
def check(yaml_config: Path, in_data: Path = None):
    """Check a YAML pipeline file for errors."""
    in_data = str(in_data) if isinstance(in_data, PurePath) else None
    return validate_yaml_config(yaml_config, in_data)


@main.command()
@click.argument("in_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "yaml_config", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "out_dir",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
)
@click.option(
    "-d",
    "--dimension",
    type=click.IntRange(1, 3),
    default=1,
    help="The dimension to slice through.",
)
@click.option(
    "--pad",
    type=click.INT,
    default=0,
    help="The number of slices to pad each block of data.",
)
@click.option(
    "--ncore",
    type=click.INT,
    default=1,
    help=" The number of the CPU cores per process.",
)
@click.option(
    "--gpu-id",
    type=click.INT,
    default=-1,
    help="The GPU ID of the device to use.",
)
@click.option(
    "--save-all",
    is_flag=True,
    help="Save intermediate datasets for all tasks in the pipeline.",
)
@click.option(
    "--file-based-reslice",
    default=None,
    is_flag=True,
    help="Reslice using intermediate files (default is in-memory).",
)
@click.option(
    "--reslice-dir",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    default=None,
    callback=lambda context, param, value: value
    if value
    else context.params["out_dir"],
    help="Directory for reslice intermediate files (defaults to out_dir, only relevant if --reslice is also given)",
)
@click.option(
    "--output-folder",
    type=click.Path(exists=False, file_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optionally define the name of the output folder created by HTTomo",
)
def run(
    in_file: Path,
    yaml_config: Path,
    out_dir: Path,
    dimension: int,
    pad: int,
    ncore: int,
    gpu_id: int,
    save_all: bool,
    file_based_reslice: bool,
    reslice_dir: Path,
    output_folder: Path,
):
    """Run a processing pipeline defined in YAML on input data."""
    # Define httomo.globals.run_out_dir in all MPI processes
    if output_folder is None:
        httomo.globals.run_out_dir = out_dir.joinpath(
            f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
        )
    else:
        httomo.globals.run_out_dir = out_dir.joinpath(output_folder)
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
        pass  # silently pass and run the CPU pipeline

    return run_tasks(
        in_file,
        yaml_config,
        dimension,
        pad,
        ncore,
        save_all,
        reslice_dir if file_based_reslice else None,
    )
