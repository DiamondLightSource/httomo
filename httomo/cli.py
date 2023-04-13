from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

import click

import httomo.globals
from httomo.common import PipelineTasks
from httomo.task_runner import run_tasks
from httomo.logger import setup_logger

from mpi4py import MPI
from . import __version__


@dataclass(frozen=True)
class GlobalOptions:
    """An immutable store of global program options."""

    in_file: Path
    yaml_config: Path
    out_dir: Path
    dimension: int
    pad: int
    ncore: int
    save_all: bool
    reslice: Optional[Path]


@click.group(invoke_without_command=True)
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
    "--save_all",
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
@click.version_option(version=__version__, message="%(version)s")
@click.pass_context
def main(
    ctx: click.Context,
    in_file: Path,
    yaml_config: Path,
    out_dir: Path,
    dimension: int,
    pad: int,
    ncore: int,
    save_all: bool,
    file_based_reslice: bool,
    reslice_dir: Path,
):
    """httomo: High Throughput Tomography."""
    ctx.obj = GlobalOptions(
        in_file,
        yaml_config,
        out_dir,
        dimension,
        pad,
        ncore,
        save_all,
        reslice_dir if file_based_reslice else None,
    )
    # Define httomo.globals.run_out_dir in all MPI processes
    httomo.globals.run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # Setup global logger object
        httomo.globals.logger = setup_logger(httomo.globals.run_out_dir)

    if ctx.invoked_subcommand is None:
        click.echo(main.get_help(ctx))


@main.command("task_runner")
@click.pass_obj
def task_runner(global_options: GlobalOptions):
    """Run the processing pipeline defined in the given YAML config file."""
    return run_tasks(
        global_options.in_file,
        global_options.yaml_config,
        global_options.dimension,
        global_options.pad,
        global_options.ncore,
        global_options.save_all,
        global_options.reslice,
    )
