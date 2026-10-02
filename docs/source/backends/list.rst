.. _backends_list:

Processing libraries
====================

HTTomo coordinates loading, processing and reconstruction methods supplied by
scientific software libraries. A pipeline may combine methods from several
:term:`backends <backend>`.

.. list-table::
   :header-rows: 1
   :widths: 20 12 31 37

   * - Library
     - Processor
     - Choose it for
     - Notes
   * - `HTTomolibGPU`_
     - GPU
     - Accelerated preprocessing, artefact correction and reconstruction
     - Uses CuPy and CUDA; several reconstruction methods also use TomoBAR or
       ASTRA.
   * - `HTTomolib`_
     - CPU
     - CPU processing and image-output methods maintained for HTTomo
     - Useful in CPU pipelines and for output stages following GPU processing.
   * - `TomoPy`_
     - CPU
     - Established tomography preprocessing and reconstruction methods
     - CPU methods exposed through HTTomo can use local multithreading within
       each MPI process.

Only methods listed in :ref:`reference_templates` have the metadata and YAML
templates required for direct use in an HTTomo pipeline. Not every function in
a backend library is exposed.

HTTomolibGPU
------------

HTTomolibGPU is developed by the Data Analysis Group at
`Diamond Light Source`_ for GPU-accelerated tomography. Its methods can be
called independently, but HTTomo adds pipeline orchestration, block sizing,
distributed I/O and GPU memory management. See :ref:`reconstruction_ecosystem`
for the relationship between HTTomolibGPU, TomoBAR, ASTRA and CuPy.

HTTomolib
---------

HTTomolib contains CPU methods maintained alongside HTTomo. It also supplies
utilities such as rescaling and image writing that are commonly used at the
end of both CPU and GPU pipelines.

TomoPy
------

TomoPy is an open-source tomography package from the wider imaging community.
HTTomo exposes a selected set of its CPU preprocessing and reconstruction
functions. HTTomo distributes data between MPI processes; TomoPy may also use
CPU threads inside each process.

Adding another method or library
--------------------------------

Processing-code integration belongs in the developer guide. See
:ref:`developers_add_own_method` for the complete method workflow and
:ref:`developers_httomo_backends` for registering metadata after a method has
been implemented.

.. _HTTomolibGPU: https://github.com/DiamondLightSource/httomolibgpu
.. _HTTomolib: https://github.com/DiamondLightSource/httomolib
.. _TomoPy: https://tomopy.readthedocs.io
.. _Diamond Light Source: https://www.diamond.ac.uk/
