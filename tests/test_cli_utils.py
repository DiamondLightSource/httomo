from pathlib import Path
import pytest
import json

from httomo.cli_utils import is_sweep_pipeline


@pytest.mark.parametrize(
    "pipeline_file, expected_is_sweep_pipeline",
    [
        ("samples/pipeline_template_examples/testing/sweep_range.yaml", True),
        ("samples/pipeline_template_examples/testing/example.yaml", False),
    ],
)
def test_is_sweep_pipeline_file(pipeline_file: Path, expected_is_sweep_pipeline: bool):
    """Test is_sweep_pipeline with file paths"""
    pipeline_file_path = Path(__file__).parent / pipeline_file
    assert is_sweep_pipeline(pipeline_file_path) is expected_is_sweep_pipeline


def test_is_sweep_pipeline_dict_with_range():
    """Test is_sweep_pipeline with a Python dict containing a sweep range"""
    pipeline_dict = '''[
        {
            "method": "normalize",
            "module_path": "httomolibgpu.prep.normalize",
            "parameters": {
                "cutoff": {
                    "start": 10,
                    "stop": 15,
                    "step": 2
                },
                "minus_log": true
            }
        }
    ]'''
    assert is_sweep_pipeline(pipeline_dict) is True


def test_is_sweep_pipeline_dict_with_list():
    """Test is_sweep_pipeline with a Python dict containing a sweep list"""
    pipeline_dict = '''[
        {
            "method": "normalize",
            "module_path": "httomolibgpu.prep.normalize",
            "parameters": {
                "cutoff": [5, 10, 15],
                "minus_log": true
            }
        }
    ]'''
    assert is_sweep_pipeline(pipeline_dict) is True


def test_is_sweep_pipeline_json_string_with_sweep():
    """Test is_sweep_pipeline with JSON strings containing both types of sweeps"""
    # Test with sweep range
    pipeline_range = '''[
        {
            "method": "normalize",
            "module_path": "httomolibgpu.prep.normalize",
            "parameters": {
                "cutoff": {
                    "start": 10,
                    "stop": 15,
                    "step": 2
                }
            }
        }
    ]'''
    assert is_sweep_pipeline(pipeline_range) is True
    
    # Test with sweep list
    pipeline_list = '''[
        {
            "method": "normalize",
            "module_path": "httomolibgpu.prep.normalize",
            "parameters": {
                "cutoff": [5, 10, 15]
            }
        }
    ]'''
    assert is_sweep_pipeline(pipeline_list) is True


def test_is_sweep_pipeline_real_world_example():
    """Test is_sweep_pipeline with a real-world pipeline JSON string"""
    # This is the exact pattern that will be encountered in real usage
    pipeline_json_str = '''
[
    {
        "method": "standard_tomo",
        "module_path": "httomo.data.hdf.loaders",
        "parameters": {
            "data_path": "auto",
            "rotation_angles": {
                "data_path": "auto"
            },
            "image_key_path": "auto"
        }
    },
    {
        "method": "normalize",
        "module_path": "httomolibgpu.prep.normalize",
        "parameters": {
            "cutoff": [
                5,
                10,
                15
            ],
            "minus_log": true,
            "nonnegativity": false,
            "remove_nans": false
        }
    },
    {
        "method": "distortion_correction_proj_discorpy",
        "module_path": "httomolibgpu.prep.alignment",
        "parameters": {
            "metadata_path": "REQUIRED",
            "order": 3,
            "mode": "constant"
        }
    },
    {
        "method": "find_center_360",
        "module_path": "httomolibgpu.recon.rotation",
        "parameters": {
            "ind": null,
            "win_width": 10,
            "side": null,
            "denoise": true,
            "norm": false,
            "use_overlap": false
        }
    },
    {
        "method": "sino_360_to_180",
        "module_path": "httomolibgpu.misc.morph",
        "parameters": {
            "overlap": "${centering.side_outputs.overlap}",
            "rotation": "left"
        }
    },
    {
        "method": "remove_stripe_based_sorting",
        "module_path": "httomolibgpu.prep.stripe",
        "parameters": {
            "size": 11,
            "dim": 1
        }
    },
    {
        "method": "FBP",
        "module_path": "httomolibgpu.recon.algorithm",
        "parameters": {
            "center": "${centering.side_outputs.centre_of_rotation}",
            "filter_freq_cutoff": 0.35,
            "recon_size": null,
            "recon_mask_radius": 0.95,
            "neglog": false
        }
    },
    {
        "method": "calculate_stats",
        "module_path": "httomo.methods",
        "parameters": {},
        "id": "statistics",
        "side_outputs": {
            "glob_stats": "glob_stats"
        }
    },
    {
        "method": "rescale_to_int",
        "module_path": "httomolibgpu.misc.rescale",
        "parameters": {
            "perc_range_min": 0,
            "perc_range_max": 100,
            "bits": 8,
            "glob_stats": "${statistics.side_outputs.glob_stats}"
        }
    },
    {
        "method": "save_to_images",
        "module_path": "httomolib.misc.images",
        "parameters": {
            "subfolder_name": "images",
            "axis": "auto",
            "file_format": "tif",
            "bits": 8,
            "perc_range_min": 0,
            "perc_range_max": 100,
            "glob_stats": "${statistics.side_outputs.glob_stats}",
            "asynchronous": true
        }
    }
]
'''
    assert is_sweep_pipeline(pipeline_json_str) is True
    
    # Also test a variation without sweep
    non_sweep_str = pipeline_json_str.replace('"cutoff": [\n                5,\n                10,\n                15\n            ]', '"cutoff": 10')
    assert is_sweep_pipeline(non_sweep_str) is False

def test_is_sweep_pipeline_full_json_with_sweep():
    """Test is_sweep_pipeline with a full pipeline JSON string with sweep range"""
    pipeline_json = '''[
        {
            "method": "standard_tomo",
            "module_path": "httomo.data.hdf.loaders",
            "parameters": {
                "data_path": "auto",
                "rotation_angles": {
                    "data_path": "auto"
                },
                "image_key_path": "auto"
            }
        },
        {
            "method": "normalize",
            "module_path": "httomolibgpu.prep.normalize",
            "parameters": {
                "cutoff": {
                    "start": 10,
                    "stop": 20,
                    "step": 5
                },
                "minus_log": true,
                "nonnegativity": false,
                "remove_nans": false
            }
        },
        {
            "method": "FBP",
            "module_path": "httomolibgpu.recon.algorithm",
            "parameters": {
                "center": "${centering.side_outputs.centre_of_rotation}",
                "filter_freq_cutoff": 0.35,
                "recon_size": null,
                "recon_mask_radius": 0.95,
                "neglog": false
            }
        }
    ]
    '''
    assert is_sweep_pipeline(pipeline_json) is True
