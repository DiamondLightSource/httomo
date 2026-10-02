.. _compatibility:

Version compatibility
=====================

Use the Python, NumPy and CuPy versions declared by the installed HTTomo
release. These packages affect the runtime interface and should not be upgraded
independently without testing the complete environment.

.. list-table:: HTTomo 3.x runtime requirements
   :header-rows: 1
   :widths: 18 20 22 20 20

   * - HTTomo release
     - Python
     - NumPy
     - CuPy
     - Pipeline archive
   * - 3.3.2
     - 3.12 or later
     - 2.4.x
     - 14.2.x
     - 3.3
   * - 3.3.0--3.3.1
     - 3.12 or later
     - 2.4.x
     - 14.0.x
     - 3.3
   * - 3.2.x
     - 3.12 or later
     - 2.4.x
     - 14.0.x
     - 3.2
   * - 3.1.x
     - 3.12 or later
     - 2.4.x
     - 14.0.x
     - 3.1

The current installation recipe uses OpenMPI 4.1.6, a parallel build of h5py,
and TomoPy 1.15.3 when TomoPy methods are required. ``mpi4py`` and h5py must be
built against compatible MPI libraries. CuPy's CUDA runtime must also be
supported by the installed NVIDIA driver.

Backend packages
----------------

HTTomo deliberately does not pin HTTomolib, HTTomolibGPU, TomoBAR or
``httomo-backends`` to one version. Their interfaces and metadata evolve
together, so treat them as one tested environment:

* use :ref:`versioned_downloads` for templates and pipelines matching a tagged
  HTTomo release;
* do not combine generated templates from ``main`` with an older release;
* rerun ``python -m httomo check PIPELINE INPUT`` after changing a backend;
* test a representative small dataset before a production run; and
* report the versions of HTTomo, ``httomo-backends`` and all processing
  libraries when requesting support.

The documentation build for this branch uses the ``httomo-backends`` version
pinned in ``docs/source/doc-pip-requirements.txt`` to generate the complete
pipeline examples. The current method reference is published separately by
``httomo-backends``. Neither resource is a universal runtime compatibility
guarantee.

Release information
-------------------

Consult the `HTTomo releases page
<https://github.com/DiamondLightSource/httomo/releases>`_ for changes and
upgrade notes. The authoritative requirements for any tag are its
``pyproject.toml`` and installation documentation.
