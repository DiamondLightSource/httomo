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

A parallel run of HTTomo at Diamond would usually be done on a compute cluster.
However, there are cases where a parallel run on a local machine on cropped data
is also useful, so that has also been described below.

TODO (:code:`httomo_mpi_local`?)
