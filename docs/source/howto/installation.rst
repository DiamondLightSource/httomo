.. _installation_main:

Installation
************

HTTomo is available from PyPI. We recommend installing it in a Conda
environment because HTTomo depends on MPI, parallel HDF5, and CUDA libraries.
A Python virtual environment can also be used when these system dependencies
are already available.

.. note:: 

   These instructions assume Linux and a CUDA-compatible GPU. For Windows or macOS, see :ref:`installation_other`.

Choose an installation path
===========================

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Platform
     - Processing support
     - Recommended path
   * - Linux with NVIDIA GPU
     - CPU and CUDA GPU methods
     - Use the Conda environment below.
   * - Linux without a GPU
     - CPU methods
     - Use the Conda environment but omit CuPy and GPU-only libraries.
   * - Windows
     - CPU and supported NVIDIA GPUs
     - Install Linux under WSL 2, then follow the Linux instructions.
   * - macOS on Apple Silicon
     - CPU methods only
     - Follow :ref:`installation_mac`.
   * - Existing managed MPI/HDF5 installation
     - Depends on local libraries
     - Use the virtual-environment path and match the locally installed MPI,
       parallel HDF5 and CUDA versions.

The commands below use Python 3.12 or later, NumPy 2.4, CuPy 14.2 and OpenMPI
4.1.6. TomoPy 1.15.3 is optional unless the selected pipeline uses TomoPy.


Conda environment
=================

.. code-block:: console

   $ conda create --name httomo --channel conda-forge \
       cupy==14.2 openmpi==4.1.6 "h5py[build=*openmpi*]" \
       python numpy astra-toolbox aiofiles click graypy loguru nvtx pillow \
       pyyaml scikit-image scipy tqdm hdf5plugin pip pywavelets
   $ conda activate httomo
   $ conda install --channel conda-forge tomopy==1.15.3  # Optional
   $ pip install --no-deps \
       httomo httomo-backends httomolib httomolibgpu tomobar


.. note::

   By default the :code:`cupy` installation will install the latest :code:`cuda-cudart`. This can result in CUDA versions higher than the supported by the GPU device of the system. One can specify the compatible to their system CUDA package, e.g., :code:`cuda-cudart==12.9.79`.


Virtual environment
===================

A Python virtual environment can be used when:

- an MPI implementation, such as OpenMPI, is installed;
- a parallel build of HDF5 is installed;
- the required CUDA libraries or CUDA Toolkit are installed; and
- TomoPy methods are not required in HTTomo pipelines.

The exact installation commands depend on the locally installed MPI, HDF5,
CUDA, and NVIDIA driver versions.

.. code-block:: console

   $ python -m venv httomo
   $ source httomo/bin/activate
   $ MPICC=$(type -p mpicc) pip install mpi4py
   $ pip install cython numpy pkgconfig setuptools # build dependencies of h5py
   $ CC=$(type -p mpicc) HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-build-isolation --no-binary=h5py h5py
   $ pip install cupy-cuda14x # install cupy-cuda14x if CUDA library/CUDA toolkit version is 14.x
   $ pip install aiofiles astra-toolbox click graypy hdf5plugin loguru nvtx pillow pyyaml scikit-image scipy tqdm
   $ pip install --no-deps httomo httomolib httomolibgpu httomo-backends tomobar

Verify the installation
=======================

Check the command-line entry point and confirm that h5py has parallel HDF5
support:

.. code-block:: console

   $ python -m httomo --version
   $ python -m httomo --help
   $ python -c "import h5py; print('Parallel HDF5:', h5py.get_config().mpi)"

The final command must print ``Parallel HDF5: True``. Developers working from
a source checkout should instead follow :ref:`developer_setup`.

.. _installation_other:

Installation on Other Platforms
===============================

.. toctree::
   :maxdepth: 2

   installation_variants/installation_windows
   installation_variants/installation_mac
