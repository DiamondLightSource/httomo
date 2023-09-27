import yaml
from pathlib import Path
from typing import Any, Dict, List, Protocol, TypeAlias
from importlib import import_module

from mpi4py import MPI

from httomo.common import LoaderInfo, MethodFunc, PreProcessInfo
from httomo.utils import Pattern, log_exception
from httomo.wrappers_class import BackendWrapper
from httomo.yaml_loader import YamlLoader


MethodConfig: TypeAlias = Dict[str, Dict[str, Any]]
PipelineStageConfig: TypeAlias = List[MethodConfig]
PipelineConfig: TypeAlias = List[PipelineStageConfig]


class PipelineReaderInterface(Protocol):
    """
    Functionalities that a pipeline YAML file reader should implement.
    """
    loader: type[YamlLoader]

    def get_loader_info(self, filepath: Path, extra_params: Dict[str, Any]) -> LoaderInfo:
        ...

    def get_pre_process_info(self, filepath: Path, comm: MPI.Comm) -> List[PreProcessInfo]:
        ...

    def get_main_pipeline_info(self, filepath: Path, comm: MPI.Comm) -> List[MethodFunc]:
        ...


class PipelineReader:
    """
    Generate necessary objects for executing the loading, pre-processing, and
    main processing stages of the pipeline.
    """

    def __init__(self, loader: type[YamlLoader]):
        loader.add_constructor("!Sweep", loader.sweep_manual)
        loader.add_constructor("!SweepRange", loader.sweep_range)
        self.loader = loader


    def get_loader_info(self, filepath: Path, extra_params: Dict[str, Any]) -> LoaderInfo:
        loader_conf: MethodConfig = self.__parse_yaml(filepath)[0][0]
        module_name, module_conf = loader_conf.popitem()
        method_name, method_conf = module_conf.popitem()
        module = import_module(module_name)
        method_func = getattr(module, method_name)
        method_conf.update(extra_params)
        return LoaderInfo(
            params=method_conf,
            method_name=method_name,
            method_func=method_func,
            pattern=Pattern.all
        )


    def get_pre_process_info(
            self,
            filepath: Path,
            comm: MPI.Comm
        ) -> List[PreProcessInfo]:
        pre_process_conf: PipelineStageConfig = self.__parse_yaml(filepath)[1]
        pre_process_infos: List[PreProcessInfo] = []
        PRE_PROCESS_METHODS = [
            "remove_outlier3d",
            "find_center_vo",
            "find_center_360"
        ]

        for task_conf in pre_process_conf:
            module_name, module_conf = task_conf.popitem()
            split_module_name = module_name.split(".")
            method_name, method_conf = module_conf.popitem()

            if method_name not in PRE_PROCESS_METHODS:
                err_str = f"Method {method_name} not allowed in pre-processing stage"
                raise ValueError(err_str)

            wrapper_init_module = BackendWrapper(
                split_module_name[0],
                split_module_name[1],
                split_module_name[2],
                method_name,
                comm
            )

            pre_process_infos.append(PreProcessInfo(
                params=method_conf,
                method_name=method_name,
                wrapper_func=wrapper_init_module.wrapper_method
            ))

        return pre_process_infos


    def get_main_pipeline_info(
            self,
            filepath: Path,
            comm: MPI.Comm
        ) -> List[MethodFunc]:
        conf: PipelineStageConfig = self.__parse_yaml(filepath)[2]
        method_funcs: List[MethodFunc] = []

        for i, task_conf in enumerate(conf):
            module_name, module_conf = task_conf.popitem()
            split_module_name = module_name.split(".")
            method_name, method_conf = module_conf.popitem()
            method_conf["method_name"] = method_name
            method_conf.pop("data_in", None)
            method_conf.pop("data_out", None)

            if split_module_name[0] not in ["tomopy", "httomolib", "httomolibgpu"]:
                err_str = (
                    f"An unknown module name was encountered: " f"{split_module_name[0]}"
                )
                log_exception(err_str)
                raise ValueError(err_str)

            wrapper_init_module = BackendWrapper(
                split_module_name[0],
                split_module_name[1],
                split_module_name[2],
                method_name,
                comm
            )
            wrapper_func = getattr(wrapper_init_module.module, method_name)
            wrapper_method = wrapper_init_module.wrapper_method

            method_funcs.append(
                MethodFunc(
                    module_name=module_name,
                    method_func=wrapper_func,
                    wrapper_func=wrapper_method,
                    parameters=method_conf,
                    cpu=True,
                    gpu=False,
                    cupyrun=False,
                    calc_max_slices=None,
                    output_dims_change=False,
                    pattern=Pattern.all,
                    return_numpy=False,
                    idx_global=i+2,
                    global_statistics=False,
                )
            )

        return method_funcs


    def __parse_yaml(self, filepath: Path) -> PipelineConfig:
        with open(filepath, "r") as f:
            yaml_conf = list(yaml.load_all(f, Loader=self.loader))
        return yaml_conf

