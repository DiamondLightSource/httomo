.. _tutorials_pl_templates:

Full YAML pipelines
==============================

This is a collection of ready to be used pipeline templates aka process lists.
See more on :ref:`explanation_process_list` and how to :ref:`howto_process_list`.

.. _tutorials_pl_templates_cpu:

CPU Pipeline templates
----------------------------

.. dropdown:: Basic TomoPy's (CPU-only) pipeline for the classical 180-degrees scan

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_cpu1.yaml

.. dropdown:: TomoPy's pipeline where :ref:`previewing` is demonstrated

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_cpu2.yaml

.. dropdown:: This pipeline shows how "calculate_stats" module extracts global statistics in order to be passed to "save_to_images" function which uses it to rescale data for saving images

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_cpu3.yaml

.. _tutorials_pl_templates_gpu:

GPU Pipeline templates
----------------------------

.. dropdown:: Basic GPU pipeline which uses functions from the httomolibgpu library.

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_gpu1.yaml


.. _tutorials_pl_templates_dls:

DLS Specific templates
----------------------------

.. dropdown:: GPU-based pipeline using httomolibgpu methods for DIAD (k11) data. Global statistics and referencing is used.

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/DLS/01_diad_pipeline_gpu.yaml

.. dropdown:: GPU-driven pipeline for the 360-degrees data which estimates the CoR value and the overlap. The 180-degrees sinogram is obtained by stitching using the overlap value. The pipeline shows the extensive use of side_outputs and refrencing.

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_360deg_gpu2.yaml

.. dropdown:: More advanced GPU pipeline for the 360-degrees data. Here we preview the section and then reconstruct it iteratively, the result then downsampled before saving smaller images.

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_360deg_iterative_gpu3.yaml

.. _tutorials_pl_templates_sweeps:

Parameter Sweeps templates
----------------------------

These templates demonstrate how to perform a sweep across multiple values of a
single parameter (see :ref:`parameter_sweeping` for more details).

.. dropdown:: Parameter sweep over 6 CoR values (`center` param) in recon
   method, and saving the result as tiffs. Note that there is need to add image saving plugin in this case. It is also preferable to keep `preview` small. 

   .. literalinclude:: ../../../tests/samples/pipeline_template_examples/parameter-sweep-cor.yaml
       :language: yaml
       :emphasize-lines: 30-33
       
.. dropdown:: Parameter sweep over 50 (`alpha` param) values of Paganin filter
   method, and saving the result as tiffs for both Paganin filter and the reconstruction module.
          
   .. literalinclude:: ../../../tests/samples/pipeline_template_examples/parameter-sweep-paganin.yaml
       :language: yaml
       :emphasize-lines: 25-28       
