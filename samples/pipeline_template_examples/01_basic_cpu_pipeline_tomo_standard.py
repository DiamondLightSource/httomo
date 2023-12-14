from datetime import datetime
from pathlib import Path
import sys
from mpi4py import MPI
import httomo
from httomo.logger import setup_logger

from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.backend_wrapper import make_backend_wrapper
from httomo.runner.loader import make_loader
from httomo.runner.output_ref import OutputRef
from httomo.runner.pipeline import Pipeline
from httomo.runner.task_runner import TaskRunner


def build_pipeline(in_file: str, comm: MPI.Comm):
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
    m_normalize = make_backend_wrapper(
        repo,
        module_path="tomopy.prep.normalize",
        method_name="normalize",
        comm=comm,
        cutoff=None,
    )
    m_minuslog = make_backend_wrapper(
        repo,
        module_path="tomopy.prep.normalize",
        method_name="minus_log",
        comm=comm,
    )
    m_center = make_backend_wrapper(
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
        output_mapping={"cor": "centre_of_rotation"},
    )
    m_recon = make_backend_wrapper(
        repo,
        module_path="tomopy.recon.algorithm",
        method_name="recon",
        comm=comm,
        center=OutputRef(m_center, "centre_of_rotation"),
        sinogram_order=False,
        algorithm="gridrec",
        init_recon=None,
    )
    m_save = make_backend_wrapper(
        repo,
        module_path="httomolib.misc.images",
        method_name="save_to_images",
        comm=comm,
        subfolder_name="images",
        axis=1,
        file_format="tif",
        bits=8,
        perc_range_min=0.0,
        perc_range_max=100.0,
        jpeg_quality=95,
    )

    return Pipeline(
        loader=loader,
        methods=[m_normalize, m_minuslog, m_center, m_recon, m_save],
        main_pipeline_start=3,  
    )
    # main pipeline:
    # essence is that if the max_slices calculation depends on an output of another method,
    # the method needs to be in a new section.


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <script> <datafile>")
        exit(1)

    out_dir = Path("httomo_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    httomo.globals.run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # Setup global logger object
        httomo.globals.logger = setup_logger(httomo.globals.run_out_dir)

    pipeline = build_pipeline(sys.argv[1], comm)
    runner = TaskRunner(pipeline, False)
    runner.execute()
    