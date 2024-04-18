.. _side_output:

Side outputs
++++++++++++

There are cases where the output dataset of a method is needed as the value of
a parameter for a method further down the pipeline. For example, the output of a
method that calculates the :ref:`centering`, that is required for a reconstruction method.

HTTomo provides a special syntax (loosely based on metadata syntax for `GitHub Actions  <https://docs.github.com/en/actions/creating-actions/metadata-syntax-for-github-actions>`_) 
how the output of the method needs to be defined and how to refer to that special output later. 

Specifying the side output
##########################

The output of some methods delivers not the processed data, but rather a supplementary information to be used later down the line. 
The given term for that supplementary data is :code:`side_outputs`. As an example, let us consider the following centering 
algorithm:

.. code-block:: yaml
  :emphasize-lines: 11,12,13

  - method: find_center_vo
    module_path: httomolibgpu.recon.rotation
    parameters:
      ind: null
      smin: -50
      smax: 50
      srad: 6.0
      step: 0.25
      ratio: 0.5
      drop: 20
    id: centering
    side_outputs:
      cor: centre_of_rotation

One can see that :code:`side_outputs` here include a singular scalar value :code:`cor` with the :code:`centre_of_rotation` reference. 
The :code:`id` parameter here needed to refer to the method later. 

Referring to the side output
############################

The purpose of :code:`side_outputs` is to refer to it later, when some method(s) require the contained information in the reference.
Consider this example then the reconstruction method refers to the centering method side outputs. The required information of :ref:`centering`
is stored in the reference :code:`${{centering.side_outputs.centre_of_rotation}}`.

.. code-block:: yaml
  :emphasize-lines: 4

  - method: FBP
    module_path: httomolibgpu.recon.algorithm
    parameters:
      center: ${{centering.side_outputs.centre_of_rotation}}
      filter_freq_cutoff: 0.6
      recon_size: null
      recon_mask_radius: null


There could be various configurations when this reference is required from other methods as well. We present more verbose :ref:`side_output_example` bellow.

.. note:: Side outputs and references to them are generated automatically with the :ref:`utilities_yamlgenerator`. Usually there is no need to modify them when you edit your process list.

.. _side_output_example:

Example of side outputs
#######################

This example demonstrates 3 cases when the side output is required and references to it. 
This pipeline is for reconstructing DFoV data which needs to be stitched into the traditional 180 degrees data. 
See that parameters for stitching are stored in side outputs and then used later in the :code:`sino_360_to_180` method.
The reconstruction module also refers to the found :ref:`centering` for the stitched dataset. Then we also need to 
extract the global statistics for normalisation of the data when saving into images. 

.. literalinclude:: ../../../../../tests/samples/pipeline_template_examples/pipeline_360deg_gpu2.yaml
  :language: yaml
  :lines: 1-71
  :emphasize-lines: 18,19,20,21,22,23,45,50,58,59,60,71
  :caption: Pipeline for 360 degrees scan, a double field of view (DFoV) data.