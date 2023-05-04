HTTomo (High Throughput Tomography pipeline)
*******************************************************

* A user interface (UI) written in Python for fast big data processing using MPI protocols
* HTTomo efficiently deals with I/O data operations while enabling processing on a CPU or a GPU
* The GPU processing can be performed purely on a device by grouping the methods together
* HTTomo can use other libraries as a backend. Currently we support `TomoPy <https://tomopy.readthedocs.io>`_ and `HTTomolib <https://github.com/DiamondLightSource/httomolib>`_
* The methods from the libraries are exposed through `YAML templates <https://github.com/DiamondLightSource/httomo/tree/main/templates>`_ enabling fast task programming

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

Running the code:
======================================================

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