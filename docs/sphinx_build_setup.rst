====================================
How to build the HTML pages locally
====================================

Create a Conda environment
==========================

Create a documentation environment from the requirements file. On a Diamond
computer, first make Conda available by loading the Python module:

.. code-block:: console

   $ module load python
   $ conda env create --name httomo-docs \
       --file /path/to/HTTomo/docs/source/doc-conda-requirements.yml
   $ conda activate httomo-docs

Alternatively, create the environment at a specific path:

.. code-block:: console

   $ conda env create --prefix /path/to/env/httomo-docs \
       --file /path/to/HTTomo/docs/source/doc-conda-requirements.yml
   $ conda activate /path/to/env/httomo-docs

Build the documentation
=======================

Run the build script from anywhere after activating the environment:

.. code-block:: console

   $ bash /path/to/HTTomo/docs/sphinx-build.sh

The script removes old generated API files and build output, then performs a
clean Sphinx HTML build. Sphinx regenerates the API summaries as part of the
build. Warnings are treated as errors, matching the documentation check in
continuous integration.

Open ``HTTomo/docs/build/index.html`` in a browser to view the result.

Check external links
====================

The scheduled continuous-integration job also checks external links. Run the
same check locally when adding or changing links:

.. code-block:: console

   $ sphinx-build -W --keep-going -a -E -b linkcheck \
       /path/to/HTTomo/docs/source /path/to/HTTomo/docs/linkcheck

Finish
======

Deactivate the environment when finished. On a Diamond computer, unload the
Python module as well:

.. code-block:: console

   $ conda deactivate
   $ module unload python
