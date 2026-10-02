.. _developer_setup:
.. _run_tests:

Development setup and testing
*****************************

Create a development checkout
=============================

Create and activate an environment using :ref:`installation_main`, then clone
HTTomo and enter the repository:

.. code-block:: console

   $ git clone https://github.com/DiamondLightSource/httomo.git
   $ cd httomo

Install the checkout and development dependencies. Choose the extra that
matches the environment:

.. code-block:: console

   $ python -m pip install --editable ".[dev-cpu]"

For a CUDA development environment, use ``.[dev-gpu]`` instead. Confirm that
Python imports the checkout and that parallel HDF5 is available:

.. code-block:: console

   $ python -c "import httomo; print(httomo.__file__)"
   $ python -c "import h5py; print('Parallel HDF5:', h5py.get_config().mpi)"

The first command should print a path inside the checkout, and the second must
print ``Parallel HDF5: True``.

Run the test suite
==================

Run commands in this section from the repository root. To test the same version
as an installed release, check out its corresponding Git tag rather than the
latest ``main`` branch.

Run the default unit-test selection:

.. code-block:: console

   $ python -m pytest tests/

Tests requiring CuPy, example datasets, performance testing, or full datasets
are skipped by default.

GPU tests
=========

On a system with a supported NVIDIA GPU and a working CuPy installation, run
the tests marked as requiring CuPy:

.. code-block:: console

   $ python -m pytest tests/ --cupy

The ``--cupy`` option selects only the CuPy tests. Run both this command and
the default test command to exercise both selections.

Small-data pipeline tests
=========================

Generate the example pipeline files from the directives installed with
``httomo-backends``:

.. code-block:: console

   $ mkdir -p docs/source/pipelines_full
   $ python docs/source/scripts/execute_pipelines_build.py \
       --output docs/source/pipelines_full/

Run the CPU/TomoPy small-data tests with:

.. code-block:: console

   $ python -m pytest tests/ --small_data

These tests require TomoPy. Tests for GPU pipelines are skipped.

On a CUDA-enabled system with CuPy, TomoBAR, and ``httomolibgpu`` installed,
run the GPU small-data tests with both selection options:

.. code-block:: console

   $ python -m pytest tests/ --small_data --cupy

Using both options selects tests marked as both ``small_data`` and ``cupy``.

Run code-quality checks
=======================

Install the repository's pre-commit hooks, then run them before opening a pull
request:

.. code-block:: console

   $ pre-commit install
   $ pre-commit run --all-files

Build the documentation
=======================

Create the documentation environment, install the same ``httomo-backends``
release pinned by the documentation workflow, generate the pipeline examples
and build with warnings treated as errors:

.. code-block:: console

   $ micromamba create --file docs/source/doc-conda-requirements.yml
   $ micromamba activate httomo-docs
   $ python -m pip install --no-deps \
       -r docs/source/doc-pip-requirements.txt
   $ python docs/source/scripts/execute_pipelines_build.py \
       --output docs/source/pipelines_full/
   $ sphinx-build -W --keep-going -a -E -b html \
       docs/source docs/build

Open ``docs/build/index.html`` to inspect the result. Update
``docs/source/doc-pip-requirements.txt`` when the complete pipeline examples
must use a new ``httomo-backends`` release; local and CI builds both consume
that file.
