.. _tutorials_pl_templates:

Process list templates
==============================

This is a collection of ready to be used pipeline templates aka process lists. 
See more on :ref:`explanation_process_list` and how to :ref:`howto_process_list`.

.. _tutorials_pl_templates_cpu:

CPU Pipeline templates
----------------------------

.. dropdown:: A basic TomoPy full data processing pipeline

    .. literalinclude:: ../../../samples/pipeline_template_examples/01_basic_cpu_pipeline_tomo_standard.yaml

.. dropdown:: Slightly more advanced TomoPy data processing pipeline with the :ref:`previewing`.
    
    .. literalinclude:: ../../../samples/pipeline_template_examples/02_basic_cpu_pipeline_tomo_standard.yaml

.. _tutorials_pl_templates_gpu:

GPU Pipeline templates
----------------------------

.. dropdown:: HTTomolib full data processing pipeline on a GPU. Note a multi-input method `remove_outlier3d` which gets to filter multiple input datasets.
    
    .. literalinclude:: ../../../samples/pipeline_template_examples/03_basic_gpu_pipeline_tomo_standard.yaml


.. _tutorials_pl_templates_dls:

DLS Specific templates
----------------------------

.. dropdown:: Diad pipeline

    .. literalinclude:: ../../../samples/pipeline_template_examples/DLS/01_diad_pipeline.yaml

.. dropdown:: 360 scan
    
    .. literalinclude:: ../../../samples/pipeline_template_examples/DLS/02_i12_360scan_pipeline.yaml

.. _tutorials_pl_templates_sweeps:

Parameter Sweeps templates
----------------------------
Those templates demonstrate how to perform sweeps across multiple values a single parameter. See more on :ref:`parameter_tuning`.

.. dropdown:: Manual variation of the Centre of Rotation in a range using :code:`!SweepRange` functionality inside the reconstruction module.

    .. literalinclude:: ../../../samples/pipeline_template_examples/parameter_sweeps/01_recon_cor_range_sweep.yaml

.. dropdown:: Manual variation of several values using :code:`Sweep` functionality inside a filter.

    .. literalinclude:: ../../../samples/pipeline_template_examples/parameter_sweeps/02_median_filter_kernel_sweep.yaml

.. dropdown:: Using :code:`!SweepRange` functionality to change a parameter in the filter inside the pipeline. Note that the results (tiff files) are saved into different folders with respect to each parameter in the sweep. 

    .. literalinclude:: ../../../samples/pipeline_template_examples/parameter_sweeps/04_phase_retrieve_image_saver.yaml
