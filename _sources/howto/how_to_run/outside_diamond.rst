.. _howto_run_outside_diamond:

Outside Diamond
+++++++++++++++

Make sure to activate the conda environment that has HTTomo installed in it.

Serial
######

This is the simplest case:

.. code-block:: console

  $ python -m httomo run IN_FILE YAML_CONFIG OUT_DIR

Parallel
########

HTTomo's parallel processing capability has been implemented with :code:`mpi4py`
and :code:`h5py`. Therefore, HTTomo is intended to be run in parallel by using
the :code:`mpirun` command (or equivalent, such as :code:`srun` for SLURM
clusters):

.. code-block:: console

  $ mpirun -np N python -m httomo run IN_FILE YAML_CONFIG OUT_DIR

where :code:`N` is the number of parallel processes to launch.

