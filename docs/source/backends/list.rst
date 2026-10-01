.. _backends_list:

Processing libraries
====================

HTTomo coordinates loading, processing and reconstruction methods supplied by
several scientific software libraries. A pipeline may combine methods from
more than one library.

Choose libraries according to the available hardware and the methods required
by the pipeline:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Library
     - Processor
     - Typical use
   * - HTTomolibgpu
     - GPU
     - GPU-accelerated preprocessing and reconstruction methods maintained for
       HTTomo.
   * - HTTomolib
     - CPU
     - CPU implementations maintained for HTTomo.
   * - TomoPy
     - CPU
     - Established CPU preprocessing and reconstruction methods from the
       wider tomography community.

Only methods listed in :ref:`reference_templates` have HTTomo YAML templates
and can be selected directly in a pipeline.

HTTomolibgpu library (GPU)
--------------------------
`HTTomolibgpu <https://github.com/DiamondLightSource/httomolibgpu>`_ library is developed at `Diamond Light source  <https://www.diamond.ac.uk/>`_
by Data Analysis Group to work together with the HTTomo software.

* HTTomolibgpu is a Python library of GPU accelerated methods written using `CuPy <https://cupy.dev/>`_ API and CUDA language.
* Most of the original methods have been taken from TomoPy or `Savu <https://github.com/DiamondLightSource/Savu>`_ software and then re-optimised and GPU-accelerated.
* Its methods can also be used independently, although HTTomo's GPU memory
  management is only available when they run as part of an HTTomo pipeline.

HTTomolib library (CPU)
--------------------------
`HTTomolib <https://github.com/DiamondLightSource/httomolib>`_ library is similar to HTTomolibgpu, but contains mostly CPU modules.

TomoPy software (CPU)
---------------------
`TomoPy <https://tomopy.readthedocs.io>`_ is an open-source Python package for
tomographic data processing and image reconstruction developed at
`The Advanced Photon Source <https://www.aps.anl.gov/>`_ in Illinois, USA.
The project is active since 2013 and it gained a `large audience <https://github.com/tomopy/tomopy>`_
of users and contributors across tomographic imaging community.

* TomoPy is an open-source package in Python and C for data processing and reconstruction. TomoPy is mostly a CPU processing library and in HTTomo we expose the CPU modules only.
* It is a CPU-multithreaded package. HTTomo controls parallelisation through MPI on a higher level and also supports local CPU multithreading from TomoPy, for every MPI process.
* Not every TomoPy function is exposed by HTTomo. See
  :ref:`reference_templates` for the supported methods.

Adding another method or library
--------------------------------

Information about integrating processing code belongs to the developer guide.
See :ref:`developers_add_own_method` for method requirements, wrappers and YAML
template generation.
