from dataclasses import dataclass
from pathlib import Path

import click

from httomo.common import PipelineTasks
from httomo.task_runner import run_tasks

from ._version_git import __version__


@dataclass(frozen=True)
class GlobalOptions:
    """An immutable store of global program options."""

    in_file: Path
    yaml_config: Path
    out_dir: Path
    dimension: int
    pad: int
    ncores: int
    save_all: bool


@click.group(invoke_without_command=True)
@click.argument("in_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "yaml_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
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
    "--ncores",
    type=click.INT,
    default=1,
    help=" The number of the CPU cores per process.",
)
@click.option(
    "--save_all",
    is_flag=True,
    help="Save intermediate datasets for all tasks in the pipeline."
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
    ncores: int,
    save_all: bool
):
    """httomo: High Throughput Tomography."""
    ctx.obj = GlobalOptions(
        in_file, yaml_config, out_dir, dimension, pad, ncores, save_all
    )

    if ctx.invoked_subcommand is None:
        click.echo(main.get_help(ctx))


@main.command('task_runner')
@click.pass_obj
def task_runner(global_options: GlobalOptions):
    """Run the processing pipeline defined in the given YAML config file.
    """
    run_tasks(
        global_options.in_file,
        global_options.yaml_config,
        global_options.out_dir,
        global_options.dimension,
        global_options.pad,
        global_options.ncores,
        global_options.save_all
    )
