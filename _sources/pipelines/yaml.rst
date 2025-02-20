.. _tutorials_pl_templates:

Full YAML pipelines
==============================

This is a collection of ready to be used full pipelines or process lists for HTTomo.
See more on :ref:`explanation_process_list` and how to :ref:`howto_process_list`.

HTTomo mainly targets GPU computations, therefore the use of :ref:`tutorials_pl_templates_gpu` is 
preferable. However, when the GPU device is not available or a GPU method is not implemented, the use of 
:ref:`tutorials_pl_templates_cpu` is possible. 

.. note:: The combination of both GPU and CPU methods is possible. If one expects to achieve the faster performance, please use the GPU methods provided, where possible.

.. _tutorials_pl_templates_gpu:

GPU Pipeline templates
-----------------------

The GPU-pipelines consist of methods from httomolibgpu (GPU) and httomolib (CPU) backend :ref:`backends_list`. Those libraries are supported directly by the HTTomo development team.

.. dropdown:: GPU pipeline with auto-centering and the FBP reconstruction method.

    .. literalinclude:: ../pipelines_full/gpu_pipelineFBP.yaml
        :language: yaml

.. dropdown:: GPU pipeline as above and Total Variation denoising on the result of the FBP reconstruction.

    .. literalinclude:: ../pipelines_full/gpu_pipelineFBP_denoising.yaml
        :language: yaml

.. _tutorials_pl_templates_cpu:

CPU Pipeline templates
-----------------------

The CPU-pipelines mostly use TomoPy methods. They are executed solely on the CPU and therefore expected to be slower than the GPU pipelines.

.. dropdown:: CPU pipeline using auto-centering and the gridrec reconstruction method on the CPU (TomoPy).

    .. literalinclude:: ../pipelines_full/cpu_pipeline_gridrec.yaml
        :language: yaml


.. _tutorials_pl_templates_dls:

DLS-specific templates
----------------------

Those pipelines are specific to Diamond Light Source processing strategies and can vary between different tomographic beamlines. 

.. dropdown:: An example of DIAD-k11 beamline pipeline with auto-centering and FBP reconstruction on the GPU.

    .. literalinclude:: ../pipelines_full/gpu_diad_FBP.yaml
        :language: yaml

.. dropdown:: Pipeline for 360-degrees data with automatic CoR/overlap finding and stitching to 180-degrees data.

    .. literalinclude:: ../pipelines_full/gpu_360_paganin_FBP.yaml
        :language: yaml

.. _tutorials_pl_templates_sweeps:

Parameter Sweeps templates
--------------------------

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
