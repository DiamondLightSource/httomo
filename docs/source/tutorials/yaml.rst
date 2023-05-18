.. _tutorials_pl_templates_cpu:

CPU Pipeline template examples
==============================
1. A basic TomoPy full data processing pipeline
--------------------------------------------------------------------
.. literalinclude:: ../../../samples/pipeline_template_examples/01_basic_cpu_pipeline_tomo_standard.yaml

2. Slightly more advanced TomoPy data processing pipeline with the :ref:`previewing`.
-------------------------------------------------------------------------------------

.. literalinclude:: ../../../samples/pipeline_template_examples/02_basic_cpu_pipeline_tomo_standard.yaml

GPU Pipeline template examples
==============================

1. HTTomolib full data processing pipeline on a GPU. Note a multi-input method `remove_outlier3d` which gets to filter multiple input datasets.
.. literalinclude:: ../../../samples/pipeline_template_examples/03_basic_gpu_pipeline_tomo_standard.yaml

Multi Inputs
-------------------------------------

.. literalinclude:: ../../../samples/pipeline_template_examples/multi_inputs/01_dezing_multi_inputs.yaml

DLS Specific
-------------------------------------

.. literalinclude:: ../../../samples/pipeline_template_examples/DLS/01_diad_pipeline.yaml

.. literalinclude:: ../../../samples/pipeline_template_examples/DLS/02_i12_360scan_pipeline.yaml