from datetime import datetime
from pathlib import Path
import sys
from mpi4py import MPI
import httomo
from httomo.logger import setup_logger

from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.backend_wrapper import BackendWrapper, make_backend_wrapper

from httomo.runner import loader
from httomo.runner.output_ref import OutputRef
from httomo.runner.pipeline import Pipeline
from httomo.runner.task_runner import TaskRunner

repo = MethodDatabaseRepository()
comm = MPI.COMM_WORLD


def make_method(module_path: str, method_name: str, **kwargs) -> BackendWrapper:
    return make_backend_wrapper(
        repo, module_path=module_path, method_name=method_name, comm=comm, **kwargs
    )


def make_loader(module_path: str, method_name: str, **kwargs) -> loader.Loader:
    return loader.make_loader(
        repo, module_path=module_path, method_name=method_name, comm=comm, **kwargs
    )


def build_pipeline(in_file: str):
    loader = make_loader(
        "httomo.data.hdf.loaders",
        "standard_tomo",
        name="tomo",
        data_path="entry1/tomo_entry/data/data",
        image_key_path="entry1/tomo_entry/instrument/detector/image_key",
        dimension=1,
        preview=[dict(), dict(start=30, stop=60), dict()],
        pad=0,
        in_file=in_file,
    )
    m_center = make_method(
        "tomopy.recon.rotation",
        "find_center_vo",
        ind="mid",
        smin=-50,
        smax=50,
        srad=6,
        step=0.25,
        ratio=0.5,
        drop=20,
        output_mapping={"cor": "center_value"},
    )
    m_normalize = make_method(
        "tomopy.prep.normalize", "normalize", cutoff=None, averaging="mean"
    )
    m_minus_log = make_method("tomopy.prep.normalize", "minus_log")
    m_remove_stripe = make_method(
        "tomopy.prep.stripe",
        "remove_stripe_fw",
        level=None,
        wname="db5",
        sigma=2,
        pad=True,
    )
    m_recon = make_method(
        "tomopy.recon.algorithm",
        "recon",
        center=OutputRef(m_center, "center_value"),
        sinogram_order=False,
        algorithm="gridrec",
        init_recon=None,
    )
    m_filter = make_method("tomopy.misc.corr", "median_filter", size=3, axis=0)
    m_save = make_method(
        "httomolib.misc.images",
        "save_to_images",
        subfolder_name="images",
        axis=0,
        file_format="tif",
        bits=8,
        perc_range_min=0.0,
        perc_range_max=100.0,
        jpeg_quality=95,
    )
    return Pipeline(
        loader=loader,
        methods=[
            m_center,
            m_normalize,
            m_minus_log,
            m_remove_stripe,
            m_recon,
            m_filter,
            m_save,
        ],
        main_pipeline_start=2,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <script> <datafile>")
        exit(1)

    out_dir = Path("httomo_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    httomo.globals.run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )

    if comm.rank == 0:
        # Setup global logger object
        httomo.globals.logger = setup_logger(httomo.globals.run_out_dir)

    pipeline = build_pipeline(sys.argv[1])
    runner = TaskRunner(pipeline, False)
    runner.execute()
