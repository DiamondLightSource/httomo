.. _choose_pipeline:

Choose a pipeline
=================

Start with the pipeline that most closely matches the hardware, scan geometry
and reconstruction goal. Then update its loader paths and processing
parameters for the input data. The table below describes the examples on
:ref:`tutorials_pl_templates`; it is guidance rather than a performance
ranking.

.. list-table::
   :header-rows: 1
   :widths: 22 11 14 24 29

   * - Starting pipeline
     - Hardware
     - Scan or mode
     - Best starting point for
     - Main dependencies
   * - ``FBP3d_tomobar``
     - GPU
     - 180°, analytical
     - Routine reconstruction with centring and stripe removal
     - HTTomolibGPU, TomoBAR and HTTomolib
   * - ``titaren_center_pc_FBP3d_resample``
     - GPU
     - 180°, analytical
     - Phase-correlation centring and downsampled output
     - HTTomolibGPU, TomoBAR and HTTomolib
   * - ``LPRec3d_tomobar``
     - GPU
     - 180°, analytical
     - Trying LPRec3d on compatible parallel-beam data
     - HTTomolibGPU, TomoBAR and HTTomolib
   * - ``FBP3d_tomobar_denoising``
     - GPU
     - 180°, analytical
     - FBP followed by total-variation denoising
     - HTTomolibGPU, TomoBAR and HTTomolib
   * - ``FISTA3d_tomobar``
     - GPU
     - 180°, iterative
     - Noisy or undersampled data where regularisation is useful
     - HTTomolibGPU, TomoBAR and HTTomolib
   * - ``deg360_paganin_FBP3d_tomobar``
     - GPU
     - 360°, analytical
     - Overlap finding, conversion to 180° and Paganin filtering
     - HTTomolibGPU, TomoBAR and HTTomolib
   * - ``deg360_distortion_FBP3d_tomobar``
     - GPU
     - 360°, analytical
     - Optical-distortion correction before 360° conversion
     - HTTomolibGPU, TomoBAR and HTTomolib
   * - ``tomopy_gridrec``
     - CPU
     - 180°, analytical
     - A small CPU run or a system without a CUDA-capable GPU
     - TomoPy and HTTomolib
   * - Sweep examples
     - GPU
     - Parameter search
     - Comparing centre-of-rotation or Paganin values
     - HTTomolibGPU, TomoBAR and HTTomolib

Before running an example:

#. Confirm that its libraries are installed; see :ref:`backends_list`.
#. Replace loader paths or use NXtomo automatic discovery; see
   :ref:`loading_data`.
#. Review method parameters against :ref:`reference_templates`.
#. Validate the result with ``python -m httomo check PIPELINE INPUT``.
#. Use the archives in :ref:`versioned_downloads` for a tagged HTTomo release.

Choose analytical reconstruction for a fast baseline. Iterative reconstruction
is more computationally expensive but can be valuable for noisy or incomplete
data. Actual speed depends on data dimensions, parameters, hardware, process
count and storage, so benchmark representative data rather than relying on a
general ranking.
