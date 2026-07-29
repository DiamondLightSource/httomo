.. _howto_run_at_diamond:

Inside Diamond
++++++++++++++

If you're not familiar what the module system is, please check :ref:`faq_workstation` guide.

Cluster
#######

This will be the most common way to use HTTomo at Diamond, it submits a job to the
production compute cluster at Diamond that will run HTTomo.

In a terminal, the commands to log onto the compute cluster and submit an HTTomo
job are the following:

.. code-block:: console

  $ ssh wilson
  $ module load httomo
  $ httomo_mpi IN_FILE YAML_CONFIG OUT_DIR

Workstation
###########

Serial
~~~~~~

HTTomo can be loaded on a Diamond workstation by doing :code:`module load httomo`.
This will allow HTTomo to be run on the local machine like so:

.. code-block:: console

  $ httomo run IN_FILE YAML_CONFIG OUT_DIR

Parallel
~~~~~~~~

Parallel execution of HTTomo at Diamond is typically performed on a compute cluster. 
The HTTomo launcher is integrated with the SLURM workload manager via REST APIs and is 
configured to submit jobs to the :code:`wilson` compute cluster.

To run HTTomo, either log in to a Wilson compute node directly or load the HTTomo environment on a workstation
with :code:`module load httomo`.

Once the environment is loaded, HTTomo jobs can be submitted and executed in parallel as described below.

.. code-block:: console

  $ httomo_mpi IN_FILE YAML_CONFIG OUT_DIR
`
Also look for help with :code:`httomo_mpi --help`.