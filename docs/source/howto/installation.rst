Installation Guide
******************

HTTomo is available on PyPI, so it can be installed into either a virtual environment or a
conda environment.

However, there are certain constraints under which a virtual environment can be used, due to
the dependence on an MPI implementation, the hdf5 library, CUDA libraries, and whether the user
requires using :code:`tomopy` methods in pipelines.

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
   $ pip install cupy-cuda12x # install cupy-cuda11x if CUDA library/CUDA toolkit version is 11.x
   $ pip install aiofiles astra-toolbox ccpi-regularisation-cupy click graypy hdf5plugin loguru nvtx pillow pyyaml scikit-image scipy tomobar tqdm
   $ pip install --no-deps httomo httomolib httomolibgpu httomo-backends

Conda environment
=================

.. code-block:: console

   $ conda env create --name httomo
   $ conda activate httomo
   $ conda install -c conda-forge cupy==12.3.0 openmpi==4.1.6 h5py[build=*openmpi*]
   $ conda install -c conda-forge tomopy==1.15 # optional
   $ pip install httomo httomolib httomolibgpu

Setup HTTomo development environment:
======================================================
.. code-block:: console

   $ pip install -e .[dev] # development mode
