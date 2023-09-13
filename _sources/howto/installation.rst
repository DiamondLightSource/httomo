Installation Guide
******************

Install HTTomo as a pre-built conda Python package
======================================================
.. code-block:: console

   $ conda env create --name httomo # create a fresh conda environment
   $ conda install -c conda-forge -c https://conda.anaconda.org/httomo/ httomo

Install as a Python module
======================================================
.. code-block:: console
    
   $ git clone git@github.com:DiamondLightSource/HTTomo.git # clone the repo
   $ conda env create --name httomo --file conda/environment.yml # install dependencies
   $ conda activate httomo # activate environment
   $ pip install .[httomolib,tomopy] # Install the module + backend(s)

Setup HTTomo development environment:
======================================================
.. code-block:: console

   $ pip install -e .[dev] # development mode 

Build HTTomo as a conda Python package
======================================================
.. code-block:: console

   $ conda build conda/recipe/ -c conda-forge -c https://conda.anaconda.org/httomo/ 
