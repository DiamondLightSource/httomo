.. _tutorials_pl_templates:

Full YAML pipelines
==============================

This is a collection of ready to be used full pipelines or process lists for HTTomo.
See more on :ref:`explanation_process_list` and how to :ref:`howto_process_list`.

HTTomo mainly targets GPU computations, therefore the use of :ref:`tutorials_pl_templates_gpu` is 
preferable. However, when the GPU device is not available or a GPU method is not implemented, the use of 
:ref:`tutorials_pl_templates_cpu` is possible. 

.. _tutorials_pl_templates_gpu:

Pipelines using HTTomo libraries
--------------------------------

Those pipelines consist of methods from HTTomolibgpu (GPU) and HTTomolib (CPU) backends :ref:`backends_list`. Those libraries are supported directly by the HTTomo development team and pipelines are built in computationally efficient way. 

.. dropdown:: Using :code:`find_center_vo` auto-centering and :code:`FBP3d_tomobar` reconstruction method, then save the result into images.

    .. literalinclude:: ../pipelines_full/FBP3d_tomobar.yaml
        :language: yaml

.. dropdown:: Using :code:`find_center_pc` auto-centering, FBP reconstruction and downsampling the result before saving the images.

    .. literalinclude:: ../pipelines_full/titaren_center_pc_FBP3d_resample.yaml
        :language: yaml

.. dropdown:: Using :code:`find_center_pc` auto-centering, FBP reconstruction and downsampling the result before saving the images.

    .. literalinclude:: ../pipelines_full/titaren_center_pc_FBP3d_resample.yaml
        :language: yaml

.. dropdown:: Using :code:`LPRec3d_tomobar` reconstruction, which is the fastest from all available reconstruction methods.

    .. literalinclude:: ../pipelines_full/LPRec3d_tomobar.yaml
        :language: yaml

.. dropdown:: Applying Total Variation denoising :code:`total_variation_PD` to the result of the FBP reconstruction.

    .. literalinclude:: ../pipelines_full/FBP3d_tomobar_denoising.yaml
        :language: yaml

.. _tutorials_pl_templates_cpu:

Pipelines using TomoPy library
------------------------------

One can build CPU-only pipelines by using mostly TomoPy methods. They are expected to be slower than the pipelines above.

.. dropdown:: CPU pipeline using auto-centering and the gridrec reconstruction method from TomoPy.

    .. literalinclude:: ../pipelines_full/tomopy_gridrec.yaml
        :language: yaml


.. _tutorials_pl_templates_dls:

DLS-specific pipelines
----------------------

These pipelines are specific to Diamond Light Source processing strategies and can vary between different tomographic beamlines. 

.. dropdown:: Reconstructing 360-degrees data with automatic CoR/overlap finding and stitching to 180-degrees data. Paganin filter is applied to the data.

    .. literalinclude:: ../pipelines_full/deg360_paganin_FBP3d_tomobar.yaml
        :language: yaml

.. dropdown:: Using distortion correction module as a part of the pipeline with 360-degrees data. 

    .. literalinclude:: ../pipelines_full/deg360_distortion_FBP3d_tomobar.yaml
        :language: yaml

.. _tutorials_pl_templates_sweeps:

Pipelines with parameter sweeps
-------------------------------

Here we demonstrate how to perform a sweep across multiple values of a single parameter (see :ref:`parameter_sweeping` for more details).

.. note::  There is no need to add image saving plugin for sweep runs as it will be added automatically. It is also preferable to keep the `preview` small as the time of computation can be substantial.

.. dropdown:: Parameter sweep using the :code:`!SweepRange` tag to do a sweep over several CoR values of the :code:`center` parameter in the reconstruction method. 

   .. literalinclude:: ../pipelines_full/sweep_center_FBP3d_tomobar.yaml
       :language: yaml
       :emphasize-lines: 34-37

.. dropdown:: Parameter sweep using the :code:`!Sweep` tag over several particular values (not a range) of the :code:`alpha` parameter for the Paganin filter. 

   .. literalinclude:: ../pipelines_full/sweep_paganin_FBP3d_tomobar.yaml
       :language: yaml
       :emphasize-lines: 53-56
            
