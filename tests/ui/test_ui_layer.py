from unittest import mock
from pytest_mock import MockerFixture


import numpy as np
import pytest
from mpi4py import MPI

from httomo.ui_layer import UiLayer


def test_pipeline1_py(standard_data, python_pipeline1):
    """Testing existing python pipelines by reading them and generating Pipelines
    """

    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(python_pipeline1, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()

    assert len(pipeline._methods) == 5
    assert LayerUI.PipelineStageConfig[0]['method'] == 'standard_tomo'
    assert LayerUI.PipelineStageConfig[0]['module_path'] == 'httomo.data.hdf.loaders'
    assert LayerUI.PipelineStageConfig[1]['method'] == 'normalize'
    assert LayerUI.PipelineStageConfig[1]['module_path'] == 'tomopy.prep.normalize'
    assert LayerUI.PipelineStageConfig[2]['method'] == 'minus_log'
    assert LayerUI.PipelineStageConfig[2]['module_path'] == 'tomopy.prep.normalize'
    assert LayerUI.PipelineStageConfig[3]['method'] == 'find_center_vo'
    assert LayerUI.PipelineStageConfig[3]['module_path'] == 'tomopy.recon.rotation'
    assert LayerUI.PipelineStageConfig[3]['side_outputs'] == {'cor' : 'centre_of_rotation'}
    assert LayerUI.PipelineStageConfig[4]['method'] == 'recon'
    assert LayerUI.PipelineStageConfig[4]['module_path'] == 'tomopy.recon.algorithm'
    assert LayerUI.PipelineStageConfig[5]['method'] == 'save_to_images'
    assert LayerUI.PipelineStageConfig[5]['module_path'] == 'httomolib.misc.images'    

def test_pipeline1_py_outref(standard_data, python_pipeline1):
    """Testing OutputRef.
    """
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(python_pipeline1, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()
    
    obj = LayerUI.PipelineStageConfig[4]['parameters']['center']
    assert obj.mapped_output_name == 'centre_of_rotation'


def test_pipeline1_yaml(standard_data, yaml_pipeline1):
    """Testing existing yaml pipelines by reading them and generating Pipelines
    """

    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(yaml_pipeline1, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()

    assert len(pipeline._methods) == 5
    assert LayerUI.PipelineStageConfig[0]['method'] == 'standard_tomo'
    assert LayerUI.PipelineStageConfig[0]['module_path'] == 'httomo.data.hdf.loaders'
    assert LayerUI.PipelineStageConfig[1]['method'] == 'normalize'
    assert LayerUI.PipelineStageConfig[1]['module_path'] == 'tomopy.prep.normalize'
    assert LayerUI.PipelineStageConfig[2]['method'] == 'minus_log'
    assert LayerUI.PipelineStageConfig[2]['module_path'] == 'tomopy.prep.normalize'
    assert LayerUI.PipelineStageConfig[3]['method'] == 'find_center_vo'
    assert LayerUI.PipelineStageConfig[3]['module_path'] == 'tomopy.recon.rotation'
    assert LayerUI.PipelineStageConfig[3]['side_outputs'] == {'cor' : 'centre_of_rotation'}
    assert LayerUI.PipelineStageConfig[4]['method'] == 'recon'
    assert LayerUI.PipelineStageConfig[4]['module_path'] == 'tomopy.recon.algorithm'
    assert LayerUI.PipelineStageConfig[5]['method'] == 'save_to_images'
    assert LayerUI.PipelineStageConfig[5]['module_path'] == 'httomolib.misc.images'


def test_pipeline1_yaml_outref(standard_data, yaml_pipeline1):
    """Testing OutputRef.
    """
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(yaml_pipeline1, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()
    
    obj = LayerUI.PipelineStageConfig[4]['parameters']['center']
    assert obj.mapped_output_name == 'centre_of_rotation'