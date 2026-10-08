.. _reconstruction_ecosystem:

Reconstruction ecosystem
========================

HTTomo exposes two main reconstruction pathways: a CPU pathway through
TomoPy and a GPU pathway through HTTomolibGPU and TomoBAR. Their pipeline
entries have the same ``method`` and ``module_path`` structure, but the
packages below those entries have different responsibilities.

The ``module_path`` identifies the public function called by HTTomo. It does
not necessarily identify the library that performs every numerical operation.
For example, a pipeline calls ``httomolibgpu.recon.algorithm.FBP3d_tomobar``;
HTTomolibGPU prepares the call, while TomoBAR and ASTRA perform parts of the
reconstruction.

Pathways at a glance
--------------------

.. list-table:: CPU and GPU reconstruction pathways
   :header-rows: 1
   :widths: 14 23 12 27 24

   * - Pathway
     - Pipeline entry
     - Execution
     - Implementation
     - Main dependencies
   * - TomoPy
     - ``tomopy.recon.algorithm.recon``
     - CPU, using NumPy arrays
     - TomoPy selects the reconstruction implementation from its
       ``algorithm`` parameter; ``gridrec`` is used by the CPU example
       pipeline.
     - TomoPy and its numerical/compiled dependencies; HTTomo distributes
       work between MPI processes.
   * - TomoBAR
     - ``httomolibgpu.recon.algorithm.<method>``
     - NVIDIA GPU; most 3D methods use CuPy arrays
     - HTTomolibGPU provides the public pipeline methods. TomoBAR supplies
       direct and iterative reconstruction, using ASTRA or its CuPy Fourier
       implementation according to the method.
     - HTTomolibGPU, TomoBAR, ASTRA Toolbox, CuPy, CUDA and a compatible
       NVIDIA driver.

``httomo-backends`` supports both pathways, but it is not a numerical
reconstruction library. It provides HTTomo with method metadata, such as the
processing pattern, memory estimator, padding and output dimensions, and it
generates the :ref:`method templates <reference_templates>`.

The GPU pathway
---------------

The diagram shows how HTTomolibGPU presents one pipeline-facing API above
the GPU reconstruction components. ASTRA supplies projection and
backprojection operators, TomoBAR supplies reconstruction algorithms and
CuPy provides CUDA-compatible arrays and kernels. Not every method uses every
component: the log-polar method, for example, follows TomoBAR's Fourier/CuPy
route rather than the ASTRA route.

.. figure:: ../_static/gpu_reconstruction_httomo.svg
   :alt: HTTomolibGPU above ASTRA Toolbox, TomoBAR and CuPy. ASTRA supplies GPU projection operators, TomoBAR supplies reconstruction algorithms, and CuPy supplies device arrays and CUDA kernels.
   :width: 100%
   :align: center

   The GPU reconstruction layers used by HTTomo. The boxes link to the
   corresponding project documentation.

.. list-table:: GPU method implementations
   :header-rows: 1
   :widths: 24 14 20 42

   * - HTTomolibGPU method
     - Type
     - Data and execution model
     - Numerical implementation
   * - ``FBP2d_astra``
     - Analytical FBP
     - GPU, reconstructed slice by slice; NumPy input and output
     - TomoBAR's two-dimensional direct-method wrapper calls ASTRA's
       ``FBP_CUDA`` implementation.
   * - ``FBP3d_tomobar``
     - Analytical FBP
     - GPU volume using CuPy arrays
     - TomoBAR applies CuPy-based filtering and uses ASTRA for GPU
       backprojection.
   * - ``LPRec3d_tomobar``
     - Analytical log-polar
     - GPU volume using CuPy arrays
     - TomoBAR performs Fourier inversion on log-polar grids using CuPy; this
       computational route does not use ASTRA projection operators.
   * - ``SIRT3d_tomobar`` and ``CGLS3d_tomobar``
     - Iterative
     - GPU volume using CuPy arrays
     - TomoBAR implements the iteration and uses ASTRA's GPU forward and
       backprojection operators.
   * - ``FISTA3d_tomobar``, ``ADMM3d_tomobar`` and ``OSEM3d_tomobar``
     - Regularised iterative
     - GPU volume using CuPy arrays
     - TomoBAR combines data-fidelity iterations, ASTRA projection operators
       and CuPy regularisers.

ASTRA remains a package dependency of TomoBAR even when a particular method,
such as ``LPRec3d_tomobar``, does not use ASTRA in its computational path.
The method name therefore describes the pipeline-facing implementation, not
the complete dependency graph.

The CPU pathway
---------------

The TomoPy pathway is shorter: HTTomo passes NumPy data to
``tomopy.recon.algorithm.recon`` and the ``algorithm`` parameter selects the
TomoPy reconstruction implementation. TomoPy may use compiled CPU kernels and
local threads, while HTTomo remains responsible for MPI distribution,
pipeline ordering and I/O.

Use this pathway for a CPU-only system, a small reconstruction, or when a
TomoPy algorithm is specifically required. See the ``tomopy_gridrec`` entry
in :ref:`choose_pipeline` for a complete CPU example.

Choosing and troubleshooting a pathway
---------------------------------------

Choose the algorithm first, then confirm that the required execution stack
is available:

* use an analytical method for a fast baseline and an iterative method when
  the data or reconstruction objective needs it;
* use the TomoPy pathway when CUDA is unavailable;
* use a TomoBAR pathway for the GPU volume methods and regularisation options;
* check :ref:`compatibility` before changing HTTomo, HTTomolibGPU, TomoBAR,
  ASTRA or CuPy independently; and
* validate the finished pipeline with
  ``python -m httomo check PIPELINE INPUT`` before a production run.

For parameter names and defaults, use :ref:`reference_templates`. For the
role of the processing libraries outside reconstruction, see
:ref:`backends_list`.
