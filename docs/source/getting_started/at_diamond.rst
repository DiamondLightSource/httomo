.. _howto_run_at_diamond:

Running HTTomo at Diamond
=========================

HTTomo can be run at Diamond in two ways:

* in parallel on the ``wilson`` compute cluster, using the ``httomo_mpi``
  launcher; or
* serially on a Diamond workstation, using the ``httomo run`` command.

Parallel execution on the compute cluster is the recommended and most common
way to process tomography data at Diamond.

Commands are entered in a terminal, also called a shell or command line. Before
running HTTomo, load its software module:

.. code-block:: console

   $ module load httomo

The module configures executable and library paths for the current terminal.
List the installed versions with ``module avail httomo``. To select a specific
version, unload the current one and load the required version:

.. code-block:: console

   $ module unload httomo
   $ module load httomo/<version>

Running HTTomo in parallel
++++++++++++++++++++++++++

Parallel HTTomo jobs run on the ``wilson`` production compute cluster. The
``httomo_mpi`` launcher is integrated with the SLURM workload manager and
submits the requested processing job to the cluster.

Submitting from a Diamond workstation
#####################################

On a Diamond workstation, load the HTTomo environment if it is not already
loaded:

.. code-block:: console

   $ module load httomo

Then submit the processing job:

.. code-block:: console

   $ httomo_mpi IN_FILE YAML_CONFIG OUT_DIR

Alternatively, log in to ``wilson``, load the HTTomo module and submit the job
from there:

.. code-block:: console

   $ ssh wilson
   $ module load httomo
   $ httomo_mpi IN_FILE YAML_CONFIG OUT_DIR

The command takes the following arguments:

``IN_FILE``
   The path to the HDF5 file containing the input tomography data.

``YAML_CONFIG``
   The path to the YAML process list that defines the processing pipeline.

``OUT_DIR``
   The directory in which HTTomo will write its output.

To see the available launcher options, run:

.. code-block:: console

   $ httomo_mpi --help

Running HTTomo serially on workstation
++++++++++++++++++++++++++++++++++++++

For smaller jobs or testing pipelines, HTTomo can be run serially on a Diamond
workstation.

First, load the HTTomo environment:

.. code-block:: console

   $ module load httomo

Then run the pipeline locally:

.. code-block:: console

   $ httomo run IN_FILE YAML_CONFIG OUT_DIR

This command runs HTTomo on the workstation itself and does not submit a job to
the compute cluster.
