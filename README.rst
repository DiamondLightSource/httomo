High Throughput Tomography software
***********************************

HTTomo is a Python framework for high-performance tomographic data processing mainly targeting GPU-compute. 
It orchestrates distributed I/O and CPU/GPU workflows using MPI, while providing 
YAML-based access to processing methods from libraries such as
`TomoPy <https://tomopy.readthedocs.io>`_ and `HTTomolibgpu <https://github.com/DiamondLightSource/httomolibgpu>`_.

Installation
============
See detailed instructions for `installation <https://diamondlightsource.github.io/httomo/howto/installation.html>`_ .

Documentation
==============
Please check the full `documentation <https://diamondlightsource.github.io/httomo/>`_.

Running HTTomo:
================

* Install the module following any chosen `installation <https://diamondlightsource.github.io/httomo/howto/installation.html>`_ path.
* For help with the command line interface, execute :code:`python -m httomo --help`
* Choose the existing `YAML pipeline <https://diamondlightsource.github.io/httomo/pipelines/yaml.html>`_ or build a new one using ready-to-be-used `templates <https://diamondlightsource.github.io/httomo/backends/templates.html>`_.
* Optional: perform the validity check of the YAML pipeline file with the `YAML checker <https://diamondlightsource.github.io/httomo/utilities/yaml_checker.html>`_.
* Run HTTomo with :code:`python -m httomo run [OPTIONS] IN_DATA_FILE YAML_CONFIG OUT_DIR`, see more on that `here <https://diamondlightsource.github.io/httomo/howto/run_httomo.html>`_.

Release Tagging Scheme
======================

We use the `setuptools-git-versioning <https://setuptools-git-versioning.readthedocs.io/en/stable/index.html>`_
package for automatically determining the version from the latest git tag.
For this to work, release tags should start with a :code:`v` followed by the actual version,
e.g. :code:`v1.1.0a`.
