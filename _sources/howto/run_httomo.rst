.. _howto_run:

How to run HTTomo
-----------------

For those interested in getting straight to the run commands, please feel free
to :ref:`jump ahead<run-commands>`. Otherwise, carry on reading for more
in-depth information on running HTTomo.

The Basics of Running HTTomo
============================

Required inputs
+++++++++++++++

In order to run HTTomo you require a data file (an HDF5 file) and a YAML process
list file that describes the desired processing pipeline. For information on
getting started creating this YAML file, please see :ref:`howto_process_list`.

Interacting with HTTomo through the command line interface (CLI)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The way to interact with the HTTomo software is through its "command line
interface" (CLI).

The preliminary step to accessing installed HTTomo software depends on if you
are using a Diamond machine or not:

- not on a Diamond machine: activate the conda environment that HTTomo was
  installed into (please refer to :doc:`installation` for instructions on how to
  install HTTomo)

- on a Diamond machine: run the command :code:`module load httomo`

Once the appropriate step has been done, you will have access to the HTTomo CLI:

.. code-block:: console

    $ python -m httomo --help
    Usage: python -m httomo [OPTIONS] COMMAND [ARGS]...

      httomo: High Throughput Tomography.

    Options:
      --version  Show the version and exit.
      --help     Show this message and exit.

    Commands:
      check  Check a YAML pipeline file for errors.
      run    Run a processing pipeline defined in YAML on input data.

As can be seen from the output above, there are two HTTomo commands available:
:code:`check` and :code:`run`.

The :code:`check` command is used for checking a YAML process list file for
errors, and is highly recommended to be run before attempting to run the
pipeline. Please see :ref:`utilities_yamlchecker` for more information about
the checks being performed, the help information that is printed, etc.

The :code:`run` command is used for running HTTomo with a pipeline on the given
HDF5 input data.

Both commands have arguments that are necessary to provide, arguments that are
optional, as well as several options/flags to customise their behaviour.

Condensed information regarding the arguments that the commands take, as well as
the options for both commands, can be found directly from the command line by
using the :code:`--help` flag, such as :code:`python -m httomo check --help`.

However, the next sections will describe each command in more detail, providing
supplementary material to the information in the CLI.

.. note:: Diamond users will be able to use :code:`httomo` as a shortcut for
          :code:`python -m httomo`

The :code:`check` command
+++++++++++++++++++++++++

.. code-block:: console

    $ python -m httomo check --help
    Usage: python -m httomo check [OPTIONS] YAML_CONFIG [IN_DATA]

      Check a YAML pipeline file for errors.

    Options:
      --help  Show this message and exit.

Arguments
#########

For :code:`check`, there is one *required* argument :code:`YAML_CONFIG`, and one
*optional* argument :code:`IN_DATA`.

:code:`YAML_CONFIG` (required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the filepath to the YAML process list file that is to be checked.

:code:`IN_DATA` (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the filepath to the HDF5 input data that you are intending to run the
YAML process list file on.

This is useful to provide because the configuration of the loader in the YAML
process list file will have some references to the internal paths within the
HDF5 file, which must be typed correctly otherwise HTTomo will fail to access
the intended dataset within the HDF5 file.

Providing the filepath to the HDF5 input data will perform a check of the loader
configuration in the YAML process list, determining if the paths mentioned in it
exist or not in the accompanying HDF5 file.

Options/flags
#############

The :code:`check` command has *no* options/flags.

The :code:`run` command
+++++++++++++++++++++++

.. code-block:: console

    $ python -m httomo run --help
    Usage: python -m httomo run [OPTIONS] IN_FILE YAML_CONFIG OUT_DIR

      Run a processing pipeline defined in YAML on input data.

    Options:
      -d, --dimension INTEGER RANGE  The dimension to slice through.  [1<=x<=3]
      --pad INTEGER                  The number of slices to pad each block of
                                     data.
      --ncore INTEGER                The number of the CPU cores per process.
      --save-all                     Save intermediate datasets for all tasks in
                                     the pipeline.
      --file-based-reslice           Reslice using intermediate files (default is
                                     in-memory).
      --reslice-dir DIRECTORY        Directory for reslice intermediate files
                                     (defaults to out_dir, only relevant if
                                     --reslice is also given)
      --help                         Show this message and exit.

Arguments
#########

For :code:`run`, there are three *required* arguments:

- :code:`IN_FILE`
- :code:`YAML_CONFIG`
- :code:`OUT_DIR`

and zero *optional* arguments.

:code:`IN_FILE` (required)
~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the filepath to the HDF5 input data that you are intending to process.

:code:`YAML_CONFIG` (required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the filepath to the YAML process list file that contains the desired
processing pipeline.

:code:`OUT_DIR` (required)
~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the path to a directory which HTTomo will create its output directory
inside.

The output directory created by HTTomo contains a date and timestamp in the
following format: :code:`{DAY}-{MONTH}-{YEAR}_{HOUR}_{MIN}_{SEC}_output/`. For
example, the output directory created for an HTTomo run on 1st May 2023 at
15:30:45 would be :code:`01-05-2023_15_30_45_output/`. If the :code:`OUT_DIR`
path provided was :code:`/home/myuser/`, then the absolute path to the output
directory created by HTTomo would be
:code:`/home/myuser/01-05-2023_15_30_45_output/`.

Options/flags
#############

The :code:`run` command has 6 options/flags:

- :code:`-d/--dimension`
- :code:`--pad`
- :code:`--ncore`
- :code:`--save-all`
- :code:`--file-based-reslice`
- :code:`--reslice-dir`

:code:`-d/--dimension`
~~~~~~~~~~~~~~~~~~~~~~

This allows the user to specify what dimension of the data (counting from 1 to
3) that the input data should be sliced in when it is first loaded by the loader
method. In other words, it allows the user to specify if the input data should
be loaded as projections (pass a value of 1, which is the default value) or
sinograms (pass a value of 2).

For example, if the method immediately after the loader processes *projections*,
then the :code:`-d/--dimension` flag can be omitted entirely, or a value of 1
can be explicitly provided like :code:`-d 1`.

Another example: if the method immediately after the loader processes
*sinograms*, then this flag needs to be passed and given a value of 2, like
:code:`-d 2`.

:code:`--pad`
~~~~~~~~~~~~~

TODO

:code:`--ncore`
~~~~~~~~~~~~~~~

In the backends that HTTomo supports, there are CPU methods which support
running multiple processes to enable the method's processing to be performed
faster.

Based on the hardware that HTTomo will be run on, the number of available CPU
cores can be provided to take advantage of this multi-process capability.

.. _httomo-saving:

:code:`--save-all`
~~~~~~~~~~~~~~~~~~

Regarding the output of methods, HTTomo's default behaviour is to *not* write
the output of a method to a file in the output directory unless one of the
following conditions is satisfied:

- the method is the last one in the processing pipeline
- the :code:`save_result` parameter has been provided a value of :code:`True` in
  a method's YAML configuration (see :ref:`save-result-examples` for more info
  on the :code:`save_result` parameter)

However, there are certain cases such as debugging, where saving the output of
all methods to files in the output directory is beneficial. This flag is a quick
way of doing so.

:code:`--file-based-reslice`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please see the :ref:`pl_reslice` section for more information about the
re-slicing operation that can occur during the execution of the processing
pipeline.

By default, HTTomo will perform the re-slice operation *without* writing a file
to the output directory, and instead perform the operation "in-memory". This is
because the latter has much better performance than the former, and is thus
given preference.

While performing the re-slice operation via writing a file has worse performance
than in-memory, it is useful to have it as an option for backup. Therefore, this
flag is for specifying to HTTomo that any re-slice operations should be done
with a file, rather than with RAM.

:code:`--reslice-dir`
~~~~~~~~~~~~~~~~~~~~~

This is related to the :code:`--file-based-reslice` flag.

By default, the directory that the file being used for the re-slice operation is
the output directory that HTTomo creates.

If this output directory is on a network-mounted disk, then read/write
operations to such a disk will in general be much slower compared to a local
disk. In particular, this means that the re-slice operation will be much slower
if the output directory is on a network-mounted disk rather than on a local
disk.

This flag can be used to specify a different directory inside which the file
used for re-slicing should reside.

In particular, if performing the re-slice with a file and the output directory is
on a *network-mounted disk*, it is recommended to use this flag to choose an
output directory that is on a *local disk* where possible. This will
*drastically* improve performance, compared to performing the re-slice with a
file on a network-mounted disk.

.. note:: If running HTTomo across multiple machines, using a single local disk
          to contain the file used for re-slicing is not possible.

Below is a summary of the different re-slicing approaches and their relative
performances:

============================ =========
Re-slice type                 Speed
============================ =========
In-memory                    Very fast
File w/ local disk           Fast
File w/ network-mounted disk Very slow
============================ =========

.. _run-commands:

Run Commands
============

As HTTomo was developed at the Diamond Light Source, there have been some extra
efforts to accommodate the users at Diamond (for example, aliases for commands
and launcher scripts). As such, there are some differences as to how one would run
HTTomo at Diamond vs. outside of Diamond, and the guidance on running HTTomo has
been split into two sections accordingly.

Additionally, HTTomo is able to run in serial or in parallel depending on what
computer hardware is available to the user, so each section has been further
split into these two subsections.

Outside Diamond
+++++++++++++++

As mentioned earlier, make sure to activate the conda environment that has
HTTomo installed in it.

Serial
######

This is the simplest case:

.. code-block:: console

  python -m httomo run IN_FILE YAML_CONFIG OUT_DIR

Parallel
########

HTTomo's parallel processing capability has been implemented with :code:`mpi4py`
and :code:`h5py`. Therefore, HTTomo is intended to be run in parallel by using
the :code:`mpirun` command (or equivalent, such as :code:`srun` for SLURM
clusters):

.. code-block:: console

  mpirun -np N python -m httomo run IN_FILE YAML_CONFIG OUT_DIR

where :code:`N` is the number of parallel processes to launch.

Inside Diamond
++++++++++++++

Serial
######

As mentioned earlier, HTTomo can be loaded on a Diamond machine by doing
:code:`module load httomo`. This will allow HTTomo to be run on the local
machine like so:

.. code-block:: console

  httomo run IN_FILE YAML_CONFIG OUT_DIR

Parallel
########

A parallel run of HTTomo at Diamond would usually be done on a compute cluster.
However, there are cases where a parallel run on a local machine on cropped data
is also useful, so that has also been described below.

Cluster
~~~~~~~

.. code-block:: console

  ssh wilson
  module load httomo
  httomo_mpi IN_FILE YAML_CONFIG OUT_DIR

Non-cluster
~~~~~~~~~~~

TODO (:code:`httomo_mpi_local`?)
