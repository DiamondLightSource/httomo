.. _side_output:

Side outputs
++++++++++++

There are cases where the output dataset of a method is needed as the value of
a parameter for a method further down the pipeline. For example, the output of a
method that calculates the :ref:`centering`, that is required for a reconstruction method.

HTTomo provides a special syntax how the output of the method needs to be defined and 
how to refer to that special output. 

Specifying the side output
##########################

The output of some methods provides a supplementary information to be used later down the line,
hence the given term for that is :code:`side_outputs`. As an example, let us consider the following centering 
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

One can see that :code:`side_outputs` here are presented by the singular scalar value :code:`cor` with the :code:`centre_of_rotation` reference. 
Please also note the :code:`id` parameter, which is a reference to the method itself. 

Refering to the side output
###########################

The sole purpose of :code:`side_outputs` is to refer to them later when some method(s) require them. There could be various combinations when this is needed
and we will present more verbose :ref:`side_output_example` bellow. Consider this reference example to the centering side outputs. 

.. code-block:: yaml
  :emphasize-lines: 4

  - method: FBP
    module_path: httomolibgpu.recon.algorithm
    parameters:
      center: ${{centering.side_outputs.centre_of_rotation}}
      filter_freq_cutoff: 0.6
      recon_size: null
      recon_mask_radius: null


.. note:: Side outputs and references to them are generated automatically with the :ref:`utilities_yamlgenerator`. Usually there is no need to modify them.

.. _side_output_example:

Example of side outputs
#######################

.. literalinclude:: ../../../../../tests/samples/pipeline_template_examples/pipeline_360deg_gpu2.yaml
  :language: yaml
  :lines: 1-71
  :emphasize-lines: 18,19,20,21,22,23,45,50,58,59,60,71
  :caption: tests/samples/pipeline_template_examples/pipeline_360deg_gpu2.yaml