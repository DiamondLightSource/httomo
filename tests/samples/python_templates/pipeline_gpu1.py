from typing import Any, Dict, List, TypeAlias

MethodConfig: TypeAlias = Dict[str, Dict[str, Any]]
PipelineStageConfig: TypeAlias = List[MethodConfig]
PipelineConfig: TypeAlias = List[PipelineStageConfig]

# NOTE: when creating a Pythonic pipeline, please use
# the function's name "methods_to_list" so it will be
# found by the loader


def methods_to_list() -> PipelineConfig:
    """Pythonic way to build a list of tasks
    from which Pipeline can be generated in Httomo.
    This script accompaniments the YAML pipeline.

    Returns:
        PipelineConfig: A list of tasks in "full_pipeline_list" that can be executed in Httomo.
    """
    full_pipeline_list = []
    loader = {
        "method": "standard_tomo",
        "module_path": "httomo.data.hdf.loaders",
        "parameters": {
            "name": "tomo",
            "data_path": "entry1/tomo_entry/data/data",
            "image_key_path": "entry1/tomo_entry/instrument/detector/image_key",
            "rotation_angles": {"data_path": "/entry1/tomo_entry/data/rotation_angle"},
            "dimension": 1,
            "pad": 0,
        },
    }
    full_pipeline_list.append(loader)
    method1 = {
        "method": "find_center_vo",
        "module_path": "httomolibgpu.recon.rotation",
        "id": "centering",
        "parameters": {
            "ind": "mid",
            "smin": -50,
            "smax": 50,
            "srad": 6,
            "step": 0.25,
            "ratio": 0.5,
            "drop": 20,
        },
        "side_outputs": {"cor": "centre_of_rotation"},
    }
    full_pipeline_list.append(method1)
    method2 = {
        "method": "remove_outlier",
        "module_path": "httomolibgpu.misc.corr",
        "parameters": {
            "dif": 0.1,
            "kernel_size": 3,
        },
    }
    full_pipeline_list.append(method2)
    method3 = {
        "method": "normalize",
        "module_path": "httomolibgpu.prep.normalize",
        "parameters": {
            "cutoff": 10.0,
            "minus_log": True,
            "nonnegativity": False,
            "remove_nans": False,
        },
    }
    full_pipeline_list.append(method3)
    method4 = {
        "method": "remove_stripe_based_sorting",
        "module_path": "httomolibgpu.prep.stripe",
        "parameters": {
            "size": 11,
            "dim": 1,
        },
    }
    full_pipeline_list.append(method4)
    method5 = {
        "method": "FBP",
        "module_path": "httomolibgpu.recon.algorithm",
        "parameters": {
            "center": "${{centering.side_outputs.centre_of_rotation}}",
            "recon_size": None,
            "recon_mask_radius": None,
        },
        "save_result": True,
    }
    full_pipeline_list.append(method5)
    method6 = {
        "method": "rescale_to_int",
        "module_path": "httomolibgpu.misc.rescale",
        "parameters": {
            "perc_range_min": 0.0,
            "perc_range_max": 100.0,
            "bits": 8,
        },
    }
    full_pipeline_list.append(method6)
    method7 = {
        "method": "save_to_images",
        "module_path": "httomolib.misc.images",
        "parameters": {
            "subfolder_name": "images",
            "axis": 1,
            "file_format": "tif",
            "jpeg_quality": 95,
        },
    }
    full_pipeline_list.append(method7)

    return full_pipeline_list
