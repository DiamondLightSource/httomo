.. _tutorials_pl_templates:

Ready-to-use pipelines
======================

This is a collection of complete HTTomo pipelines, also called process lists.
Use one as a starting point and adapt its loader paths and method parameters to
the input data. See :ref:`explanation_process_list` for the underlying concepts
and :ref:`howto_process_list` for configuration guidance.

HTTomo primarily targets GPU processing, so use
:ref:`tutorials_pl_templates_gpu` when a compatible GPU is available. Otherwise,
select a :ref:`tutorials_pl_templates_cpu` pipeline.

.. warning::
   These examples track the current HTTomo development version. For production
   with a tagged release, use the matching pipeline from
   :ref:`versioned_downloads`.

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

.. dropdown:: Using :code:`LPRec3d_tomobar` reconstruction, which is the fastest from all available reconstruction methods.

    .. literalinclude:: ../pipelines_full/LPRec3d_tomobar.yaml
        :language: yaml

.. dropdown:: Applying Total Variation denoising :code:`total_variation_PD` to the result of the FBP reconstruction.

    .. literalinclude:: ../pipelines_full/FBP3d_tomobar_denoising.yaml
        :language: yaml

.. dropdown:: Using advanced iterative reconstruction :code:`FISTA3d_tomobar` with Total Variation regularisation. Recommended for undersampled and/or noisy data.

    .. literalinclude:: ../pipelines_full/FISTA3d_tomobar.yaml
        :language: yaml

.. _tutorials_pipelines:

Tutorial pipelines
------------------

These pipelines are for :ref:`data_tutorials`, where data is also provided or can be generated.

.. dropdown:: TomoPy (CPU) pipeline for :ref:`real-data-lorentz`

    .. literalinclude:: ../pipelines_full/tomopy_tomobank.yaml
        :language: yaml

.. dropdown:: GPU-enabled processing for :ref:`real-data-lorentz`

    .. literalinclude:: ../pipelines_full/FBP3d_tomobar_tomobank.yaml
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

.. note::  There is no need to add image saving plugin for sweep runs as it will be added automatically. 

.. dropdown:: Parameter sweep using the :code:`!SweepRange` tag to do a sweep over several CoR values of the :code:`center` parameter in the reconstruction method. 

   .. literalinclude:: ../pipelines_full/sweep_center_FBP3d_tomobar.yaml
       :language: yaml
       :emphasize-lines: 36-39

.. dropdown:: Parameter sweep using the :code:`!Sweep` tag over several particular values (not a range) of the :code:`ratio_delta_beta` parameter for the Paganin filter. 

   .. literalinclude:: ../pipelines_full/sweep_paganin_FBP3d_tomobar.yaml
       :language: yaml
       :emphasize-lines: 51-54
            

.. _tutorials_pl_templates_cpu:

Pipelines using TomoPy library
------------------------------

One can build CPU-only pipelines by using mostly TomoPy methods. 

.. note::  Methods from TomoPy are expected to be slower than the GPU-accelerated methods from the libraries above.

.. dropdown:: CPU pipeline using auto-centering and the gridrec reconstruction method from TomoPy.

    .. literalinclude:: ../pipelines_full/tomopy_gridrec.yaml
        :language: yaml
