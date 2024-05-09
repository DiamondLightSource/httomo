from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from pathlib import Path, PurePath
from shutil import copy
import sys
import tempfile
from typing import List, TextIO, Union

import click
from mpi4py import MPI
from loguru import logger

import httomo.globals
from httomo.logger import setup_logger
from httomo.monitors import MONITORS_MAP, make_monitors
from httomo.transform_layer import TransformLayer
from httomo.yaml_checker import validate_yaml_config
from httomo.runner.task_runner import TaskRunner
from httomo.ui_layer import UiLayer

from . import __version__


@click.group
@click.version_option(version=__version__, message="%(version)s")
def main():
    """httomo: Software for High Throughput Tomography in parallel beam.

    Use `python -m httomo run --help` for more help on the runner.
    """
    pass


@main.command()
@click.argument(
    "yaml_config", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "in_data_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False, default=None,
)
def check(yaml_config: Path, in_data_file: Path = None):
    """Check a YAML pipeline file for errors."""
    in_data = str(in_data_file) if isinstance(in_data_file, PurePath) else None
    return validate_yaml_config(yaml_config, in_data)


@main.command()
@click.argument(
    "in_data_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
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
    help="Directory for temporary files potentially needed for reslicing (defaults to output dir)",
)
@click.option(
    "--max-cpu-slices",
    type=click.INT,
    default=64,
    help="Maximum number of slices to use for a block for CPU-only sections (default: 64)"
)
@click.option(
    "--max-memory",
    type=click.STRING,
    default="0",
    help="Limit the amount of memory used by the pipeline to the given memory (supports strings like 3.2G or bytes)"
)
@click.option(
    "--monitor",
    type=click.STRING,
    multiple=True,
    default=[],
    help=("Add monitor to the runner (can be given multiple times). " +
          f"Available monitors: {', '.join(MONITORS_MAP.keys())}")
)
@click.option(
    "--monitor-output",
    type=click.File('w'),
    default=sys.stdout,
    help="File to store the monitoring output. Defaults to '-', which denotes stdout"
)
def run(
    in_data_file: Path,
    yaml_config: Path,
    out_dir: Path,
    gpu_id: int,
    save_all: bool,
    reslice_dir: Union[Path, None],
    max_cpu_slices: int,
    max_memory: str,
    monitor: List[str],
    monitor_output: TextIO,
):
    """Run a pipeline defined in YAML on input data."""

    # we use half the memory for blocks since we typically have inputs/output
    memory_limit = transform_limit_str_to_bytes(max_memory) // 2

    if max_cpu_slices < 1:
        raise ValueError("max-cpu-slices must be greater or equal to 1")
    httomo.globals.MAX_CPU_SLICES = max_cpu_slices

    # Define httomo.globals.run_out_dir in all MPI processes
    httomo.globals.run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # Create timestamped output directory
        Path.mkdir(httomo.globals.run_out_dir, exist_ok=True)

        # Copy YAML pipeline file to output directory
        copy(yaml_config, httomo.globals.run_out_dir)

    setup_logger(Path(httomo.globals.run_out_dir))

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

    # perform transformations on pipeline
    tr = TransformLayer(comm=comm, save_all=save_all)
    pipeline = tr.transform(pipeline)

    # Run the pipeline using Taskrunner, with temp dir or reslice dir
    mon = make_monitors(monitor)
    ctx: AbstractContextManager = nullcontext(reslice_dir)
    if reslice_dir is None:
        ctx = tempfile.TemporaryDirectory()
    with ctx as tmp_dir:
        runner = TaskRunner(
            pipeline,
            Path(tmp_dir),
            monitor=mon,
            memory_limit_bytes=memory_limit,
        )
        runner.execute()
        if mon is not None:
            mon.write_results(monitor_output)


def _check_yaml(yaml_config: Path, in_data: Path):
    """Check a YAML pipeline file for errors."""
    return validate_yaml_config(yaml_config, in_data)


def transform_limit_str_to_bytes(limit_str: str):
    try:
        limit_upper = limit_str.upper()
        if limit_upper.endswith("K"):
            return int(float(limit_str[:-1]) * 1024)
        elif limit_upper.endswith("M"):
            return int(float(limit_str[:-1]) * 1024**2)
        elif limit_upper.endswith("G"):
            return int(float(limit_str[:-1]) * 1024**3)
        else:
            return int(limit_str)
    except ValueError:
        raise ValueError(f"invalid memory limit string {limit_str}")
