High Throughput Tomography software
***********************************

HTTomo is a Python framework for high-performance tomographic data processing mainly targeting GPU-compute.
It orchestrates distributed I/O and CPU/GPU workflows using MPI, while providing
YAML-based access to processing methods from libraries such as
`TomoPy <https://tomopy.readthedocs.io>`_ and `HTTomolibgpu <https://github.com/DiamondLightSource/httomolibgpu>`_.

Documentation
=============

The `HTTomo documentation <https://diamondlightsource.github.io/httomo/>`_
contains installation instructions, a quickstart, ready-to-use pipelines,
reference material and developer guidance.

After installation, use :code:`python -m httomo --help` to inspect the command
line interface. A typical workflow is to validate a pipeline and then run it:

.. code-block:: console

   python -m httomo check pipeline.yaml input.nxs
   python -m httomo run input.nxs pipeline.yaml output_directory

Release Tagging Scheme
======================

We use the `setuptools-git-versioning <https://setuptools-git-versioning.readthedocs.io/en/stable/index.html>`_
package for automatically determining the version from the latest git tag.
For this to work, release tags should start with a :code:`v` followed by the actual version,
e.g. :code:`v1.1.0a`.
