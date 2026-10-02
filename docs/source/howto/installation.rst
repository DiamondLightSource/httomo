.. _installation_main:

Installation
************

HTTomo is available from PyPI. We recommend installing it in a Conda
environment because HTTomo depends on MPI and parallel HDF5; GPU installations
also require compatible CUDA libraries. A Python virtual environment can be
used when these system dependencies are already available.

.. note::

   The primary recipe assumes Linux and a CUDA-compatible GPU. A Linux CPU-only
   recipe is also provided. For Windows or macOS, see
   :ref:`installation_other`.

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
     - Use the CPU-only Conda environment below.
   * - Windows
     - CPU and supported NVIDIA GPUs
     - Install Linux under WSL 2, then follow the Linux instructions.
   * - macOS on Apple Silicon
     - CPU methods only
     - Follow :ref:`installation_mac`.

The commands below use Python 3.12, NumPy 2.4, CuPy 14.2 and OpenMPI 4.1.6.
TomoPy 1.15.3 is optional unless the selected pipeline uses TomoPy. See
:ref:`compatibility` before changing these versions.


Conda environment with GPU support
==================================

.. code-block:: console

   $ conda create --name httomo --channel conda-forge \
       python=3.12 "numpy==2.4.*" "cupy==14.2.*" \
       openmpi==4.1.6 mpi4py "h5py[build=*openmpi*]" \
       astra-toolbox aiofiles click graypy loguru nvtx pillow pyyaml \
       scikit-image scipy tqdm hdf5plugin pip pywavelets
   $ conda activate httomo
   $ conda install --channel conda-forge tomopy==1.15.3  # Optional
   $ pip install --no-deps \
       httomo httomo-backends httomolib httomolibgpu tomobar


.. note::

   Conda may select a ``cuda-cudart`` version that is newer than the installed
   NVIDIA driver supports. If necessary, add a compatible CUDA runtime to the
   create command, for example ``cuda-cudart==12.9.79``.

.. _installation_cpu_only:

CPU-only Conda environment
==========================

Use this environment on systems without a CUDA-capable GPU. It omits CuPy,
HTTomolibGPU and TomoBAR but retains MPI and parallel HDF5:

.. code-block:: console

   $ conda create --name httomo-cpu --channel conda-forge \
       python=3.12 "numpy==2.4.*" openmpi==4.1.6 mpi4py \
       "h5py[build=*openmpi*]" astra-toolbox aiofiles click graypy loguru \
       pillow pyyaml scikit-image scipy tqdm hdf5plugin pip pywavelets
   $ conda activate httomo-cpu
   $ conda install --channel conda-forge tomopy==1.15.3
   $ pip install --no-deps httomo httomo-backends httomolib


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

   $ python3.12 -m venv httomo
   $ source httomo/bin/activate
   $ MPICC=$(type -p mpicc) pip install mpi4py
   $ pip install cython "numpy==2.4.*" pkgconfig setuptools  # h5py build dependencies
   $ CC=$(type -p mpicc) HDF5_MPI="ON" \
       HDF5_DIR=/path/to/parallel-hdf5 \
       pip install --no-build-isolation --no-binary=h5py h5py
   $ pip install "cupy-cuda14x==14.2.*"  # For a CUDA 14.x runtime
   $ pip install aiofiles astra-toolbox click graypy hdf5plugin loguru \
       nvtx pillow pyyaml scikit-image scipy tqdm
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
