Installation Guide
******************

Install HTTomo as a pre-built conda Python package
==================================================

This installation is preferable as it should take care all of dependencies for you by getting them from the dedicated anaconda channel. 

.. code-block:: console

   $ conda env create --name httomo # create a fresh conda environment
   $ conda install "httomo/linux-64::httomo * py310_openmpi_regular*" -c conda-forge -c astra-toolbox -c rapidsai

Install as a Python module
===========================

If installation above for some reason is not working for you, then the best way to install HTTomo is to create conda environment first and then 
`pip install` HTTomo into it. You will need to `git clone` HTTomo repository to your disk first. 

.. code-block:: console
    
   $ git clone git@github.com:DiamondLightSource/HTTomo.git # clone the repo
   $ conda env create --name httomo --file conda/environment.yml # install dependencies
   $ conda activate httomo # activate environment
   $ pip install . # Install the module

Setup HTTomo development environment:
======================================================
.. code-block:: console

   $ pip install -e .[dev] # development mode 

Build HTTomo as a conda Python package
======================================================
.. code-block:: console

   $ conda build conda/recipe/ -c conda-forge -c https://conda.anaconda.org/httomo/ -c rapidsai --no-test
