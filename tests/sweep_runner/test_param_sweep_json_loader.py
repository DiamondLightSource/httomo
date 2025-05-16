import numpy as np

from httomo.sweep_runner.param_sweep_json_loader import ParamSweepJsonLoader


def test_load_range_sweep():
    PARAM_NAME = "parameter_1"
    JSON_STRING = """
[
    {
        "method": "standard_tomo",
        "module_path": "httomo.data.hdf.loaders",
        "parameters": {}
    },
    {
        "method": "some_method",
        "module_path": "some.module.path",
        "parameters": {
            "parameter_1": {
                "start": 10,
                "stop": 110,
                "step": 5
            }
        }
    }
]
"""
    data = ParamSweepJsonLoader(JSON_STRING).load()
    assert isinstance(data[1]["parameters"][PARAM_NAME], tuple)
    assert data[1]["parameters"][PARAM_NAME] == tuple(np.arange(10, 110, 5))


def test_param_value_with_start_stop_step_and_other_keys_unaffected_by_range_sweep_parsing():
    PARAM_NAME = "parameter_1"
    JSON_STRING = """
[
    {
        "method": "standard_tomo",
        "module_path": "httomo.data.hdf.loaders",
        "parameters": {}
    },
    {
        "method": "some_method",
        "module_path": "some.module.path",
        "parameters": {
            "parameter_1": {
                "start": 10,
                "stop": 110,
                "step": 5,
                "another": 0
            }
        }
    }
]
"""
    data = ParamSweepJsonLoader(JSON_STRING).load()
    assert isinstance(data[1]["parameters"][PARAM_NAME], dict)
    assert data[1]["parameters"][PARAM_NAME] == {
        "start": 10,
        "stop": 110,
        "step": 5,
        "another": 0,
    }
