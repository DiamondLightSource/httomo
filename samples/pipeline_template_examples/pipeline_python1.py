import numpy as np
from typing import Any, Dict, List, TypeAlias

MethodConfig: TypeAlias = Dict[str, Dict[str, Any]]
PipelineStageConfig: TypeAlias = List[MethodConfig]
PipelineConfig: TypeAlias = List[PipelineStageConfig]

# NOTE: when creating a Pythonic pipeline, please use
# the name "methods_to_list" for the function

def methods_to_list() -> PipelineConfig:
    """Pythonic way to build a list of tasks
    from which Pipeline can be generated in Httomo.
    This accompaniments the YAML interface.

    Returns:
        PipelineConfig: A list of tasks that can be executed in Httomo.
    """
    full_pipeline = []
    loader = {
        'method': "standard_tomo",
        'module_path': "httomo.data.hdf.loaders",
        'parameters' : {
                     'name': 'tomo',
                     'data_path': 'entry1/tomo_entry/data/data',
                     'image_key_path': 'entry1/tomo_entry/instrument/detector/image_key',
                     'rotation_angles': '{"data_path": "/entry1/tomo_entry/data/rotation_angle"}',
                     'dimension': 1,
                     'preview': [dict(), dict(), dict()],
                     'pad': 0,
                       },
                    }
    full_pipeline.append(loader)
    method1 = {
        'method': "normalize",
        'module_path': "tomopy.prep.normalize",
        'parameters' : {
                     'cutoff': None,
                       },
                    }
    full_pipeline.append(method1)
    method2 = {
        'method': "minus_log",
        'module_path': "tomopy.prep.normalize",
        'parameters' : {},
                    }
    full_pipeline.append(method2)
    return full_pipeline
    