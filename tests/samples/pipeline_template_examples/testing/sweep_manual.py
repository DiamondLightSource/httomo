from httomo.ui_layer import PipelineConfig


def methods_to_list() -> PipelineConfig:
    full_pipeline_list = []
    loader = {
        "method": "standard_tomo",
        "module_path": "httomo.data.hdf.loaders",
        "parameters": {
            "data_path": "entry1/tomo_entry/data/data",
            "image_key_path": "entry1/tomo_entry/instrument/detector/image_key",
            "rotation_angles": {"data_path": "/entry1/tomo_entry/data/rotation_angle"},
        },
    }
    full_pipeline_list.append(loader)
    method1 = {
        "method": "normalize",
        "module_path": "tomopy.prep.normalize",
        "parameters": {
            "cutoff": None,
        },
    }
    full_pipeline_list.append(method1)
    sweep_method = {
        "method": "median_filter",
        "module_path": "tomopy.misc.corr",
        "parameters": {
            "size": (3, 5),
            "axis": 0,
        },
    }
    full_pipeline_list.append(sweep_method)

    return full_pipeline_list
