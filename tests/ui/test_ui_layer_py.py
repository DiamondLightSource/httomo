from unittest import mock

import numpy as np
from mpi4py import MPI

from httomo.ui_layer import UiLayer


def test_pipeline_cpu1(standard_data, python_pipeline1):
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

def test_test_pipeline_cpu1_outref(standard_data, python_pipeline1):
    """Testing OutputRef.
    """
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(python_pipeline1, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()
    
    obj = LayerUI.PipelineStageConfig[4]['parameters']['center']
    assert obj.mapped_output_name == 'centre_of_rotation'


def test_pipeline_cpu2(standard_data, python_pipeline2):
    """Testing existing python pipelines by reading them and generating Pipelines
    """

    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(python_pipeline2, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()

    assert len(pipeline._methods) == 8
    assert LayerUI.PipelineStageConfig[0]['method'] == 'standard_tomo'
    assert LayerUI.PipelineStageConfig[0]['module_path'] == 'httomo.data.hdf.loaders'
    assert LayerUI.PipelineStageConfig[0]['parameters']['preview'] == [None, {'start': 30, 'stop': 60}, None]
    assert LayerUI.PipelineStageConfig[1]['method'] == 'find_center_vo'
    assert LayerUI.PipelineStageConfig[1]['module_path'] == 'tomopy.recon.rotation'    
    assert LayerUI.PipelineStageConfig[1]['side_outputs'] == {'cor' : 'centre_of_rotation'}
    assert LayerUI.PipelineStageConfig[2]['method'] == 'remove_outlier'
    assert LayerUI.PipelineStageConfig[2]['module_path'] == 'tomopy.misc.corr'    
    assert LayerUI.PipelineStageConfig[3]['method'] == 'normalize'
    assert LayerUI.PipelineStageConfig[3]['module_path'] == 'tomopy.prep.normalize'
    assert LayerUI.PipelineStageConfig[4]['method'] == 'minus_log'
    assert LayerUI.PipelineStageConfig[4]['module_path'] == 'tomopy.prep.normalize'
    assert LayerUI.PipelineStageConfig[5]['method'] == 'remove_stripe_fw'
    assert LayerUI.PipelineStageConfig[5]['module_path'] == 'tomopy.prep.stripe'
    assert LayerUI.PipelineStageConfig[6]['method'] == 'recon'
    assert LayerUI.PipelineStageConfig[6]['module_path'] == 'tomopy.recon.algorithm'
    assert LayerUI.PipelineStageConfig[7]['method'] == 'median_filter'
    assert LayerUI.PipelineStageConfig[7]['module_path'] == 'tomopy.misc.corr'
    assert LayerUI.PipelineStageConfig[8]['method'] == 'save_to_images'
    assert LayerUI.PipelineStageConfig[8]['module_path'] == 'httomolib.misc.images'


def test_pipeline_cpu1_outref(standard_data, python_pipeline2):
    """Testing OutputRef.
    """
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(python_pipeline2, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()
    
    obj = LayerUI.PipelineStageConfig[6]['parameters']['center']
    assert obj.mapped_output_name == 'centre_of_rotation'

def test_pipeline_gpu1(standard_data, python_pipeline3):
    """Testing existing python pipelines by reading them and generating Pipelines
    """

    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(python_pipeline3, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()

    assert len(pipeline._methods) == 6
    assert LayerUI.PipelineStageConfig[0]['method'] == 'standard_tomo'
    assert LayerUI.PipelineStageConfig[0]['module_path'] == 'httomo.data.hdf.loaders'
    assert LayerUI.PipelineStageConfig[0]['parameters']['preview'] == [None, None, None]
    assert LayerUI.PipelineStageConfig[1]['method'] == 'find_center_vo'
    assert LayerUI.PipelineStageConfig[1]['module_path'] == 'httomolibgpu.recon.rotation'    
    assert LayerUI.PipelineStageConfig[1]['side_outputs'] == {'cor' : 'centre_of_rotation'}
    assert LayerUI.PipelineStageConfig[2]['method'] == 'remove_outlier3d'
    assert LayerUI.PipelineStageConfig[2]['module_path'] == 'httomolibgpu.misc.corr'    
    assert LayerUI.PipelineStageConfig[3]['method'] == 'normalize'
    assert LayerUI.PipelineStageConfig[3]['module_path'] == 'httomolibgpu.prep.normalize'
    assert LayerUI.PipelineStageConfig[4]['method'] == 'remove_stripe_based_sorting'
    assert LayerUI.PipelineStageConfig[4]['module_path'] == 'httomolibgpu.prep.stripe'
    assert LayerUI.PipelineStageConfig[5]['method'] == 'FBP'
    assert LayerUI.PipelineStageConfig[5]['module_path'] == 'httomolibgpu.recon.algorithm'
    assert LayerUI.PipelineStageConfig[6]['method'] == 'save_to_images'
    assert LayerUI.PipelineStageConfig[6]['module_path'] == 'httomolib.misc.images'


def test_pipeline_gpu1_outref(standard_data, python_pipeline3):
    """Testing OutputRef.
    """
    comm = MPI.COMM_WORLD
    LayerUI = UiLayer(python_pipeline3, standard_data, comm=comm)

    pipeline = LayerUI.build_pipeline()
    
    obj = LayerUI.PipelineStageConfig[5]['parameters']['center']
    assert obj.mapped_output_name == 'centre_of_rotation'    