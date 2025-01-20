.. _tutorials_pl_templates:

Full YAML pipelines
==============================

This is a collection of ready to be used pipeline templates aka process lists for HTTomo.
See more on :ref:`explanation_process_list` and how to :ref:`howto_process_list`.

.. _tutorials_pl_templates_cpu:

CPU Pipeline templates
----------------------------

CPU-pipelines mostly use TomoPy methods that are executed on the CPU and expected to be slower.

.. dropdown:: Basic TomoPy's (CPU-only) pipeline for the classical 180-degrees scan

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_cpu1.yaml
        :language: yaml

.. dropdown:: TomoPy's pipeline where :ref:`previewing` is demonstrated

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_cpu2.yaml
        :language: yaml

.. dropdown:: This pipeline shows how "calculate_stats" module extracts global statistics in order to rescale data for saving 8-bit images

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_cpu3.yaml
        :language: yaml

.. _tutorials_pl_templates_gpu:

GPU Pipeline templates
----------------------------

It is recommended to use GPU-based pipelines and methods from the httomolib and httomolibgpu libraries. Those libraries are supported directly by HTTomo development team. 

.. dropdown:: Basic GPU pipeline which uses functions from the httomolibgpu library.

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_gpu1.yaml
        :language: yaml


.. _tutorials_pl_templates_dls:

DLS-specific templates
----------------------------

Those pipelines will use the methods from the httomolib and httomolibgpu libraries. 

.. dropdown:: An example of a typical DIAD (k11) beamline piepeline.

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/DLS/01_diad_pipeline_gpu.yaml
        :language: yaml

.. dropdown:: Pipeline for 360-degrees data with automatic CoR finding and stitching to 180-degrees data.

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_360deg_gpu2.yaml
        :language: yaml

.. dropdown:: Pipeline for 360-degrees data with automatic CoR finding and stitching to 180-degrees data. Iterative reconstruction

    .. literalinclude:: ../../../tests/samples/pipeline_template_examples/pipeline_360deg_iterative_gpu3.yaml
        :language: yaml

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
