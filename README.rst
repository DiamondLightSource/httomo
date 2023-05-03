HTTOmo (High Throughput Tomography pipeline)
********************************************

* A Python tool for parallel read of HDF5 tomographic data using MPI protocols
* The data can be re-chunked, saved and re-loaded (e.g. projection or sinogram-wise)
* All `TomoPy <https://tomopy.readthedocs.io>`_ functions are exposed through YAML templates enabling fast task programming

Setup a Development Environment:
================================

* Clone the repository from GitHub using :code:`git clone git@github.com:DiamondLightSource/HTTomo.git`
* Install dependencies from the environment file :code:`conda env create --file conda/environment.yml` (SLOW)
* Alternatively you can install from the existing explicit file :code:`conda create --name httomo --file conda/explicit/latest.txt`
* Activate the environment with :code:`conda activate httomo`
* Install the enviroment in development mode with :code:`pip install -e .[dev]`


Install as a Python module
==========================

* You should choose which backend(s) you'd like to use - either :code:`tomopy` or :code:`httomolib`, or both
* Install the module + backend(s) with :code:`pip install .[httomolib,tomopy]`

Running the code:
=================

* Install the module as described in "Install as a Python module"
* Execute the python module with :code:`httomo <args>`
* For help with the command line interface, execute :code:`httomo --help`

Am example of running the code with test data:
==============================================

* Go to the home directory and run: :code:`python -m httomo testdata/tomo_standard.nxs . cpu`

An example of running the code with test data:
==============================================

* Create an output directory :code:`mkdir output_dir/`
* Go to the home directory and run: :code:`httomo tests/test_data/tomo_standard.nxs samples/pipeline_template_examples/02_basic_cpu_pipeline_tomo_standard.yaml output_dir/ task_runner`

Release Tagging Scheme
======================

We use the `setuptools-git-versioning <https://setuptools-git-versioning.readthedocs.io/en/stable/index.html>`_
package for automatically determining the version from the latest git tag.
For this to work, release tags should start with a :code:`v` followed by the actual version,
e.g. :code:`v1.1.0a`.
We have setup a  :code:`tag_filter` in :code:`pyproject.toml` to filter tags following this pattern.