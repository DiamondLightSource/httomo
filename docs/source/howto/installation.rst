Installation Guide
******************

HTTomo is available on PyPI, so it can be installed into either a virtual environment or a
conda environment.

However, there are certain constraints under which a virtual environment can be used, due to
the dependence on an MPI implementation, and whether the user requires using :code:`tomopy`
methods in pipelines.

Install HTTomo as a PyPi package
=========================================================
.. code-block:: console

   $ pip install httomo # this will install the CPU-only version
   $ pip install httomo httomolibgpu # this will install the GPU backend

Install as a Python module
===========================

If installation above for some reason is not working for you, then the best way to install HTTomo is to create conda environment first and then
`pip install` HTTomo into it. You will need to `git clone` HTTomo repository to your disk first.  Use `environment.yml` file to install
the GPU-supported HTTomo. For CPU-only version, please use `environment-cpu.yml` instead.

.. code-block:: console

   $ git clone git@github.com:DiamondLightSource/HTTomo.git # clone the repo
   $ conda env create --name httomo --file conda/environment.yml # install dependencies for GPU version
   $ conda activate httomo # activate environment
   $ pip install . # Install the module

Setup HTTomo development environment:
======================================================
.. code-block:: console

   $ pip install -e .[dev] # development mode
