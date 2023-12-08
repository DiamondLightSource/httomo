import yaml
from pathlib import Path
from typing import Any, Dict, List, Protocol, TypeAlias
from importlib import import_module
import sys

from mpi4py import MPI
import httomo
from httomo.logger import setup_logger

from httomo.utils import log_exception
from httomo.runner.pipeline import Pipeline


from datetime import datetime

# from httomo.common import LoaderInfo, MethodFunc, PreProcessInfo
# from httomo.utils import Pattern, log_exception
# from httomo.wrappers_class import BackendWrapper

MethodConfig: TypeAlias = Dict[str, Dict[str, Any]]
PipelineStageConfig: TypeAlias = List[MethodConfig]
PipelineConfig: TypeAlias = List[PipelineStageConfig]

def load_yaml_file(filepath: Path) -> Pipeline:
     PipelineStageConfig = __parse_yaml(filepath)
     print("boo")
     return 0


def __parse_yaml(filepath: Path) -> PipelineConfig:
    with open(filepath, "r") as f:
        yaml_conf = list(yaml.load_all(f, Loader=yaml.FullLoader))
    return yaml_conf     

# class PipelineReader:
#     """
#     Generate necessary objects for executing the loading, pre-processing, and
#     main processing stages of the pipeline.
#     """

#     def __init__(self, loader: type[YamlLoader]):
#         loader.add_constructor("!Sweep", loader.sweep_manual)
#         loader.add_constructor("!SweepRange", loader.sweep_range)
#         self.loader = loader


#     def get_loader_info(self, filepath: Path, extra_params: Dict[str, Any]) -> LoaderInfo:
#         loader_conf: MethodConfig = self.__parse_yaml(filepath)[0][0]
#         module_name, module_conf = loader_conf.popitem()
#         method_name, method_conf = module_conf.popitem()
#         module = import_module(module_name)
#         method_func = getattr(module, method_name)
#         method_conf.update(extra_params)
#         if "preview" not in method_conf.keys():
#             method_conf["preview"] = [None]
#         return LoaderInfo(
#             params=method_conf,
#             method_name=method_name,
#             method_func=method_func,
#             pattern=Pattern(extra_params["dimension"] - 1)
#         )


#     def get_pre_process_info(
#             self,
#             filepath: Path,
#             comm: MPI.Comm
#         ) -> List[PreProcessInfo]:
#         pre_process_conf: PipelineStageConfig = self.__parse_yaml(filepath)[1]
#         pre_process_infos: List[PreProcessInfo] = []
#         PRE_PROCESS_METHODS = [
#             "remove_outlier3d",
#             "find_center_vo",
#             "find_center_360"
#         ]

#         for task_conf in pre_process_conf:
#             module_name, module_conf = task_conf.popitem()
#             split_module_name = module_name.split(".")
#             method_name, method_conf = module_conf.popitem()

#             if method_name not in PRE_PROCESS_METHODS:
#                 err_str = f"Method {method_name} not allowed in pre-processing stage"
#                 raise ValueError(err_str)

#             wrapper_init_module = BackendWrapper(
#                 split_module_name[0],
#                 split_module_name[1],
#                 split_module_name[2],
#                 method_name,
#                 comm
#             )

#             pre_process_infos.append(PreProcessInfo(
#                 params=method_conf,
#                 method_name=method_name,
#                 module_path=module_name,
#                 wrapper_func=wrapper_init_module.wrapper_method
#             ))

#         return pre_process_infos


#     def get_main_pipeline_info(
#             self,
#             filepath: Path,
#             comm: MPI.Comm
#         ) -> List[MethodFunc]:
#         conf: PipelineStageConfig = self.__parse_yaml(filepath)[2]
#         method_funcs: List[MethodFunc] = []

#         for i, task_conf in enumerate(conf):
#             module_name, module_conf = task_conf.popitem()
#             split_module_name = module_name.split(".")
#             method_name, method_conf = module_conf.popitem()
#             method_conf["method_name"] = method_name

#             if split_module_name[0] not in ["tomopy", "httomolib", "httomolibgpu"]:
#                 err_str = (
#                     f"An unknown module name was encountered: " f"{split_module_name[0]}"
#                 )
#                 log_exception(err_str)
#                 raise ValueError(err_str)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <script> <yaml_file>")
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

    pipeline = load_yaml_file(sys.argv[1])
    #runner = TaskRunner(pipeline, False)
    #runner.execute()