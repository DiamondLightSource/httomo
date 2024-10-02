from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from pathlib import Path, PurePath
from shutil import copy
import sys
import tempfile
from typing import List, Optional, TextIO, Union

import click
from mpi4py import MPI
from loguru import logger

import httomo.globals
from httomo.cli_utils import is_sweep_pipeline
from httomo.logger import setup_logger
from httomo.monitors import MONITORS_MAP, make_monitors
from httomo.runner.pipeline import Pipeline
from httomo.sweep_runner.param_sweep_runner import ParamSweepRunner
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
    required=False,
    default=None,
)
def check(yaml_config: Path, in_data_file: Optional[Path] = None):
    """Check a YAML pipeline file for errors."""
    in_data = in_data_file if isinstance(in_data_file, PurePath) else None
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
    "--output-folder-name",
    type=click.Path(exists=False, file_okay=False, writable=True, path_type=Path),
    default=None,
    help="Define the name of the output folder created by HTTomo",
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
    help="Maximum number of slices to use for a block for CPU-only sections (default: 64)",
)
@click.option(
    "--max-memory",
    type=click.STRING,
    default="0",
    help="Limit the amount of memory used by the pipeline to the given memory (supports strings like 3.2G or bytes)",
)
@click.option(
    "--monitor",
    type=click.STRING,
    multiple=True,
    default=[],
    help=(
        "Add monitor to the runner (can be given multiple times). "
        + f"Available monitors: {', '.join(MONITORS_MAP.keys())}"
    ),
)
@click.option(
    "--monitor-output",
    type=click.File("w"),
    default=sys.stdout,
    help="File to store the monitoring output. Defaults to '-', which denotes stdout",
)
@click.option(
    "--intermediate-format",
    type=click.Choice(["hdf5"], case_sensitive=False),
    default="hdf5",
    help="Write intermediate data in hdf5 format",
)
@click.option(
    "--compress-intermediate",
    is_flag=True,
    help="Write intermediate data in chunked format with BLOSC compression applied",
)
@click.option(
    "--syslog-host",
    type=click.STRING,
    default="localhost",
    help="Host of the syslog server",
)
@click.option(
    "--syslog-port",
    type=click.INT,
    default=514,
    help="Port on the host the syslog server is running on",
)
@click.option(
    "--frames-per-chunk",
    type=click.IntRange(0),
    default=1,
    help="Number of frames per-chunk in intermediate data (0 = write as contiguous)",
)
def run(
    in_data_file: Path,
    yaml_config: Path,
    out_dir: Path,
    output_folder_name: Optional[Path],
    gpu_id: int,
    save_all: bool,
    reslice_dir: Union[Path, None],
    max_cpu_slices: int,
    max_memory: str,
    monitor: List[str],
    monitor_output: TextIO,
    intermediate_format: str,
    compress_intermediate: bool,
    syslog_host: str,
    syslog_port: int,
    frames_per_chunk: int,
):
    """Run a pipeline defined in YAML on input data."""
    set_global_constants(
        out_dir,
        intermediate_format,
        compress_intermediate,
        frames_per_chunk,
        max_cpu_slices,
        syslog_host,
        syslog_port,
        output_folder_name,
    )

    does_contain_sweep = is_sweep_pipeline(yaml_config)
    global_comm = MPI.COMM_WORLD
    method_wrapper_comm = global_comm if not does_contain_sweep else MPI.COMM_SELF

    if global_comm.rank == 0:
        initialise_output_directory(yaml_config)

    setup_logger(Path(httomo.globals.run_out_dir))

    pipeline = generate_pipeline(
        in_data_file, yaml_config, save_all, method_wrapper_comm
    )

    if not does_contain_sweep:
        execute_high_throughput_run(
            pipeline,
            global_comm,
            gpu_id,
            max_memory,
            monitor,
            monitor_output,
            reslice_dir,
        )
    else:
        execute_sweep_run(pipeline, global_comm)


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


def _set_gpu_id(gpu_id: int):
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


def set_global_constants(
    out_dir: Path,
    intermediate_format: str,
    compress_intermediate: bool,
    frames_per_chunk: int,
    max_cpu_slices: int,
    syslog_host: str,
    syslog_port: int,
    output_folder_name: Optional[Path],
) -> None:
    if compress_intermediate:
        frames_per_chunk = 1
    httomo.globals.INTERMEDIATE_FORMAT = intermediate_format
    httomo.globals.COMPRESS_INTERMEDIATE = compress_intermediate
    httomo.globals.FRAMES_PER_CHUNK = frames_per_chunk
    httomo.globals.SYSLOG_SERVER = syslog_host
    httomo.globals.SYSLOG_PORT = syslog_port

    if output_folder_name is None:
        httomo.globals.run_out_dir = out_dir.joinpath(
            f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
        )
    else:
        httomo.globals.run_out_dir = out_dir.joinpath(output_folder_name)

    if max_cpu_slices < 1:
        raise ValueError("max-cpu-slices must be greater or equal to 1")
    httomo.globals.MAX_CPU_SLICES = max_cpu_slices


def initialise_output_directory(yaml_config: Path) -> None:
    Path.mkdir(httomo.globals.run_out_dir, exist_ok=True)
    copy(yaml_config, httomo.globals.run_out_dir)


def generate_pipeline(
    in_data_file: Path, yaml_config: Path, save_all: bool, method_wrapper_comm: MPI.Comm
) -> Pipeline:
    # instantiate UiLayer class for pipeline build
    init_UiLayer = UiLayer(yaml_config, in_data_file, comm=method_wrapper_comm)
    pipeline = init_UiLayer.build_pipeline()

    # perform transformations on pipeline
    tr = TransformLayer(comm=method_wrapper_comm, save_all=save_all)
    pipeline = tr.transform(pipeline)

    return pipeline


def execute_high_throughput_run(
    pipeline: Pipeline,
    global_comm: MPI.Comm,
    gpu_id: int,
    max_memory: str,
    monitor: List[str],
    monitor_output: TextIO,
    reslice_dir: Union[Path, None],
) -> None:
    # we use half the memory for blocks since we typically have inputs/output
    memory_limit = transform_limit_str_to_bytes(max_memory) // 2

    _set_gpu_id(gpu_id)

    # Run the pipeline using Taskrunner, with temp dir or reslice dir
    mon = make_monitors(monitor, global_comm)
    ctx: AbstractContextManager = nullcontext(reslice_dir)
    if reslice_dir is None:
        ctx = tempfile.TemporaryDirectory()
    with ctx as tmp_dir:
        runner = TaskRunner(
            pipeline,
            Path(tmp_dir),
            global_comm,
            monitor=mon,
            memory_limit_bytes=memory_limit,
        )
        runner.execute()
        if mon is not None:
            mon.write_results(monitor_output)


def execute_sweep_run(pipeline: Pipeline, global_comm: MPI.Comm) -> None:
    ParamSweepRunner(pipeline, global_comm).execute()
