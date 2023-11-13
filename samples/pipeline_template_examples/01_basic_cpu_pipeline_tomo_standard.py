import sys
from mpi4py import MPI

from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.backend_wrapper import make_backend_wrapper
from httomo.runner.loader import make_loader
from httomo.runner.pipeline import Pipeline
from httomo.runner.task_runner import TaskRunner


def build_pipeline(in_file: str):
    comm = MPI.COMM_WORLD
    repo = MethodDatabaseRepository()
    loader = make_loader(
        repo,
        "httomo.data.hdf.loaders",
        "standard_tomo",
        comm,
        name="tomo",
        data_path="entry1/tomo_entry/data/data",
        image_key_path="entry1/tomo_entry/instrument/detector/image_key",
        dimension=1,
        preview=[dict(), dict(), dict()],
        pad=0,
        in_file=in_file,
    )
    methods = [
        make_backend_wrapper(
            repo,
            module_path="tomopy.prep.normalize",
            method_name="normalize",
            comm=comm,
            cutoff=None,
        ),
        make_backend_wrapper(
            repo,
            module_path="tomopy.prep.normalize",
            method_name="minus_log",
            comm=comm,
        ),
        make_backend_wrapper(
            repo,
            module_path="tomopy.recon.rotation",
            method_name="find_center_vo",
            comm=comm,
            ind="mid",
            smin=-50,
            smax=50,
            srad=6,
            step=0.25,
            ratio=0.5,
            drop=20,
        ),
        make_backend_wrapper(
            repo,
            module_path="tomopy.recon.algorithm",
            method_name="recon",
            comm=comm,
            center="cor",
            sinogram_order=False,
            algorithm="gridrec",
            init_recon=None,
        ),
        make_backend_wrapper(
            repo,
            module_path="httomolib.misc.images",
            method_name="save_to_images",
            comm=comm,
            subfolder_name="images",
            axis=0,
            file_format="tif",
            bits=8,
            perc_range_min=0.0,
            perc_range_max=100.0,
            jpeg_quality=95,
        ),
    ]
    return Pipeline(loader=loader, methods=methods, main_pipeline_start=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <script> <datafile>")
        exit(1)

    pipeline = build_pipeline(sys.argv[1])
    runner = TaskRunner(pipeline, False)
    runner.execute()
