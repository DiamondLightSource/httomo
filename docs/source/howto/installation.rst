Installation Guide
******************

HTTomo is available on PyPI, so it can be installed into either a virtual environment or a
conda environment.

However, there are certain constraints under which a virtual environment can be used, due to
the dependence on an MPI implementation, the hdf5 library, CUDA libraries, and whether the user
requires using :code:`tomopy` methods in pipelines.

Conda environment
=================

By default the :code:`cupy` installation will install the latest :code:`cuda-cudart`. This can result in CUDA versions higher than the supported by the GPU device of the system. One can specify the compatible to their system CUDA package, e.g., :code:`cuda-cudart==12.9.79`.

.. code-block:: console

   $ conda create --name httomo
   $ conda activate httomo
   $ conda install -c conda-forge cupy==14.0.1 openmpi==4.1.6 h5py[build=*openmpi*] python numpy astra-toolbox aiofiles click graypy loguru nvtx pillow pyyaml scikit-image scipy tqdm hdf5plugin pip pywavelets
   $ conda install -c conda-forge tomopy==1.15.3 # optional
   $ pip install httomo httomo-backends httomolib httomolibgpu tomobar --no-deps

Setup HTTomo development environment:
======================================================

Development mode requires git cloning the HTTomo's repository and pip installing from the source as bellow. Note that all other dependencies, apart from :code:`httomo`, must be satisfied as above.

.. code-block:: console

   $ pip install -e .[dev] # development mode

Virtual environment
===================

A virtual environment can be used if the following conditions are met:

- an MPI implementation is installed on the system (ie, OpenMPI)
- the hdf5 library is installed on the system
- CUDA libraries or CUDA toolkit are installed on the system
- methods from :code:`tomopy` are not required to be used in pipelines

.. code-block:: console

   $ python -m venv httomo
   $ source httomo/bin/activate
   $ MPICC=$(type -p mpicc) pip install mpi4py==3.1.6
   $ pip install cython numpy pkgconfig setuptools # build dependencies of h5py
   $ CC=$(type -p mpicc) HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-build-isolation --no-binary=h5py h5py
   $ pip install cupy-cuda13x # install cupy-cuda13x if CUDA library/CUDA toolkit version is 13.x
   $ pip install aiofiles astra-toolbox click graypy hdf5plugin loguru nvtx pillow pyyaml scikit-image scipy tqdm
   $ pip install --no-deps httomo httomolib httomolibgpu httomo-backends tomobar


macOS (Apple Silicon) — CPU-only Installation
==============================================

.. note::
   HTTomo's GPU-accelerated methods (``httomolibgpu``) depend on `CuPy
   <https://cupy.dev/>`_, which requires an NVIDIA CUDA GPU. Apple Silicon
   Macs (M1/M2/M3/M4) have no CUDA support, so this path installs HTTomo in
   **CPU-only mode**, using TomoPy for reconstruction instead of the GPU
   backends. Pipelines must use CPU/TomoPy methods only (see
   :doc:`../pipelines/yaml` for an example CPU pipeline).

This guide has been tested on an M1 MacBook (16GB RAM) running native
arm64 conda (not under Rosetta).

1. Install a native arm64 conda distribution
---------------------------------------------

Make sure you install the **arm64**, not Intel/x86_64, build — otherwise
everything below runs emulated under Rosetta and is significantly slower:

.. code-block:: bash

   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
   bash Miniforge3-MacOSX-arm64.sh

2. Create the environment
---------------------------

Skip ``cupy`` entirely — there is no arm64/macOS build, and it cannot be
installed on Apple Silicon. ``astra-toolbox`` and ``tomopy`` do have
osx-arm64 conda-forge builds (CPU-only algorithms), which is all a CPU
pipeline needs. ``mpi4py`` is required even for a single-process run, since
HTTomo's CLI unconditionally imports ``mpi4py.MPI``.
Replace `conda` with `mamba` below if it's available in the environment, for a faster package resolution.

.. code-block:: bash

   conda create --name httomo python=3.11
   conda activate httomo

   # numpy must stay below 2.0 — HTTomo's CPU/GPU array-type detection
   # (block.is_gpu) relies on numpy.ndarray *not* having a `.device`
   # attribute, an assumption NumPy 2.0's Array API support breaks.
   conda install -c conda-forge "numpy<2" mpi4py openmpi==4.1.6 \
     "h5py=*=mpi_openmpi*" astra-toolbox tomopy==1.15.3 \
     aiofiles click graypy loguru nvtx pillow pyyaml \
     scikit-image scipy tqdm hdf5plugin pywavelets

   # compilers needed to build httomolib's OpenMP-based C extension —
   # macOS's system clang has no -fopenmp support
   conda install -c conda-forge compilers llvm-openmp

3. Install HTTomo
-------------------

.. code-block:: bash

   pip install httomo httomo-backends httomolib tomobar --no-deps

Verify:

.. code-block:: bash

   python -m httomo --help

4. Known issues on this path (as of httomo 3.0 / httomolib 4.0.1)
---------------------------------------------------------------------

If ``h5py`` ever gets silently swapped back to a non-MPI build by a later ``conda install`` (check with
``python -c "import h5py; print(h5py.get_config().mpi)"``), pin it:

  .. code-block:: bash

     conda config --env --append pinned_packages 'h5py=*=mpi_openmpi*'

5. Running a CPU pipeline
---------------------------

Always validate the pipeline first:

.. code-block:: bash

   python -m httomo check pipeline.yaml data.nxs

Run serially:

.. code-block:: bash

   python -m httomo run data.nxs pipeline.yaml ./output --max-memory 10G

Or across multiple CPU cores with MPI (``--max-memory`` is per-process,
so divide your budget by the process count):

.. code-block:: bash

   mpirun -np 4 python -m httomo run data.nxs pipeline.yaml ./output --max-memory 2G

.. note::
   With 16GB of unified memory shared with macOS itself, keep
   ``--max-memory`` well under the physical total (8–10G total budget is a
   safe starting point) to avoid swapping.
