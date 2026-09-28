.. _run_tests:

Test an HTTomo installation
***************************

Basic verification
==================

To verify that HTTomo is installed in the active environment, run:

.. code-block:: console

   $ python -m httomo --version
   $ python -m httomo --help
   $ python -c "import httomo; print(httomo.__version__)"
   $ python -c "import h5py; print('Parallel HDF5:', h5py.get_config().mpi)"

The final command should print ``Parallel HDF5: True``.

Run the test suite
==================

The test suite is intended primarily for development and requires an HTTomo
source checkout. Testing a checkout is different from testing an installed
wheel because Python imports HTTomo from the repository while the tests run.

Clone the repository and enter its root directory:

.. code-block:: console

   $ git clone https://github.com/DiamondLightSource/httomo.git
   $ cd httomo

.. note:: 

   To test the same version as an installed release, check out its corresponding Git tag rather than testing the latest ``main`` branch.


Install the test dependencies into the active environment:

.. code-block:: console

   $ conda install --channel conda-forge pytest pytest-mock plumbum

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
