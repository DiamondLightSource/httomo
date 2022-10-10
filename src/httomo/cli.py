from dataclasses import dataclass
from pathlib import Path

import click

from httomo.common import PipelineTasks
from httomo.cpu_pipeline import cpu_pipeline
from httomo.gpu_pipeline import gpu_pipeline

from ._version_git import __version__


@dataclass(frozen=True)
class GlobalOptions:
    """An immutable store of global program options."""

    in_file: Path
    out_dir: Path
    data_key: str
    dimension: int
    crop: int
    pad: int
    stop_after: PipelineTasks


@click.group(invoke_without_command=True)
@click.argument("in_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "out_dir",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
)
@click.option(
    "-k",
    "--data_key",
    type=click.STRING,
    default="/entry1/tomo_entry/data/data",
    help="The path to data within the hdf5 file.",
)
@click.option(
    "-d",
    "--dimension",
    type=click.IntRange(1, 3),
    default=1,
    help="The dimension to slice through.",
)
@click.option(
    "--crop",
    type=click.IntRange(1, 100),
    default=100,
    help="The percentage of data to process.",
)
@click.option(
    "--pad",
    type=click.INT,
    default=0,
    help="The number of slices to pad each block of data.",
)
@click.option(
    "--stop_after",
    type=click.Choice(PipelineTasks._member_names_, False),
    callback=lambda c, p, v: PipelineTasks[str(v).upper()]
    if v is not None
    else PipelineTasks.SAVE,
    help="Stop after the specified stage.",
)
@click.version_option(version=__version__, message="%(version)s")
@click.pass_context
def main(
    ctx: click.Context,
    in_file: Path,
    out_dir: Path,
    data_key: str,
    dimension: int,
    crop: int,
    pad: int,
    stop_after: PipelineTasks,
):
    """httomo: High Throughput Tomography."""
    ctx.obj = GlobalOptions(
        in_file, out_dir, data_key, dimension, crop, pad, stop_after
    )

    if ctx.invoked_subcommand is None:
        click.echo(main.get_help(ctx))


@main.command()
@click.pass_obj
def cpu(global_options: GlobalOptions):
    """Perform reconstruction using the reference CPU pipeline."""
    cpu_pipeline(
        global_options.in_file,
        global_options.out_dir,
        global_options.data_key,
        global_options.dimension,
        global_options.crop,
        global_options.pad,
        global_options.stop_after,
    )


@main.command()
@click.pass_obj
def gpu(global_options: GlobalOptions):
    """Perform reconstruction using the GPU accelerated pipeline."""
    gpu_pipeline(
        global_options.in_file,
        global_options.out_dir,
        global_options.data_key,
        global_options.dimension,
        global_options.crop,
        global_options.pad,
        global_options.stop_after,
    )
