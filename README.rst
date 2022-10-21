HTTOmo (High Throughput Tomography pipeline)
******************************************

* A Python tool for parallel read of HDF5 tomographic data using MPI protocols
* The data can be re-chunked, saved and re-loaded (e.g. projection or sinogram-wise)
* All `TomoPy <https://tomopy.readthedocs.io>`_ functions are exposed through YAML templates enabling fast task programming

Setup a Development Environment:
================================
* Clone the repository from GitHub using :code:`git clone git@github.com:DiamondLightSource/HTTomo.git`
* Install dependencies from the environment file :code:`conda env create httomo --file conda/environment.yml` (SLOW)
* Alternatively you can install from the existing explicit file :code:`conda create --name httomo --file conda/explicit/latest.txt`
* Activate the environment with :code:`conda activate httomo`
* Install the enviroment in development mode with :code:`python setup.py develop`

Install as a Python module
==========================

* Ensure all necessary dependencies are present in the environment
* Install the module with :code:`python setup.py install`

Running the code:
=================

* Install the module as described in "Install as a Python module"
* Execute the python module with :code:`python -m httomo <args>`
* For help with the command line interface, execute :code:`python -m httomo --help`

Am example of running the code with test data:
=================
* Go to the home directory and run:
* :code:`python -m httomo testdata/tomo_standard.nxs . cpu`