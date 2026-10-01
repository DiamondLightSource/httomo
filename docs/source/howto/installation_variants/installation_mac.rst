.. _installation_mac:

macOS (Apple Silicon)
*********************

.. note::

   HTTomo's GPU-accelerated methods (``httomolibgpu``) depend on `CuPy
   <https://cupy.dev/>`_, which requires an NVIDIA CUDA GPU. Apple Silicon
   Macs (M1/M2/M3/M4) have no CUDA support, so this path installs HTTomo in
   **CPU-only mode**, using TomoPy for reconstruction instead of the GPU
   backends. Pipelines must use CPU/TomoPy methods only (see
   :ref:`tutorials_pl_templates` for an example CPU pipeline).

This guide has been tested on an M1 MacBook (16GB RAM) running native
arm64 conda (not under Rosetta).

Installation steps
==================

1. Install a native arm64 conda distribution

Make sure you install the **arm64**, not Intel/x86_64, build — otherwise
everything below runs emulated under Rosetta and is significantly slower:

.. code-block:: bash

   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
   bash Miniforge3-MacOSX-arm64.sh

2. Create the environment

HTTomo requires Python 3.12 or later and NumPy 2.4. CuPy and
``httomolibgpu`` are not installed because they require an NVIDIA CUDA GPU.

``mpi4py`` is required even for a single-process run because HTTomo imports
``mpi4py.MPI`` when its command-line interface starts.

.. code-block:: console

   $ conda create --name httomo --channel conda-forge \
       python=3.12 "numpy==2.4.*" \
       mpi4py openmpi==4.1.6 "h5py=*=mpi_openmpi*" \
       tomopy==1.15.3 astra-toolbox \
       aiofiles click graypy loguru nvtx pillow pyyaml \
       scikit-image scipy tqdm hdf5plugin pywavelets \
       compilers llvm-openmp pip
   $ conda activate httomo

NumPy 2.x is required by the current HTTomo implementation. In particular,
HTTomo uses the ``numpy.ndarray.device`` attribute introduced in NumPy 2.0 to
identify CPU arrays.

The ``compilers`` and ``llvm-openmp`` packages are needed when building
HTTomoLib's OpenMP-based extension because the system Clang compiler supplied
by macOS does not provide OpenMP support by default.

3. Install HTTomo

Install only the CPU backend packages. ``--no-deps`` is required because the
published package metadata currently includes CUDA-only dependencies that
cannot be installed on Apple Silicon.

.. code-block:: console

   $ python -m pip install --no-deps \
       httomo httomo-backends httomolib

Do not install ``httomolibgpu`` or ``tomobar`` in this environment. Both are
GPU-oriented packages with CUDA dependencies.

4. Verify the installation

Confirm the Python and NumPy versions and verify that parallel HDF5 is enabled:

.. code-block:: console

   $ python -c "import sys, numpy; print(sys.version); print(numpy.__version__)"
   $ python -c "import h5py; print('Parallel HDF5:', h5py.get_config().mpi)"
   $ python -m httomo --help

The first command should report Python 3.12 or later and NumPy 2.4.x. The
second command should print ``Parallel HDF5: True``.

Developers who need to run the source test suite should follow
:ref:`developer_setup`.
