.. _howto_run:

How to run HTTomo
-----------------

The next section gives an overview of the commands to quickly get started running
HTTomo.

For those interested in learning about the different ways HTTomo can be configured
to run, there is the :ref:`run-httomo-indepth` section.

Quick Overview of Running HTTomo
================================

Required inputs
+++++++++++++++

In order to run HTTomo you require a data file (an HDF5 file) and a YAML process
list file that describes the desired processing pipeline. For information on
getting started creating this YAML file, please see :ref:`howto_process_list`
and also ready-to-be-used :ref:`tutorials_pl_templates`.

Running HTTomo Inside or Outside of Diamond
+++++++++++++++++++++++++++++++++++++++++++

As HTTomo was developed at the Diamond Light Source, there have been some extra
efforts to accommodate the users at Diamond (for example, aliases for commands
and launcher scripts). As such, there are some differences as to how one would run
HTTomo at Diamond vs. outside of Diamond, and the guidance on running HTTomo has
been split into two sections accordingly.

Additionally, HTTomo is able to run in serial or in parallel depending on what
computer hardware is available to the user, so some sections have been further
split into these two subsections where relevant.

.. toctree::
   :maxdepth: 2

   how_to_run/at_diamond
   how_to_run/outside_diamond

.. _run-httomo-indepth:

In-depth Look at Running HTTomo
===============================

Interacting with HTTomo through the command line interface (CLI)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The way to interact with the HTTomo software is through its "command line
interface" (CLI).

As mentioned earlier, the preliminary step to accessing installed HTTomo software
depends on if you are using a Diamond machine or not:

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
    Usage: python -m httomo run [OPTIONS] IN_DATA_FILE YAML_CONFIG OUT_DIR

      Run a pipeline defined in YAML on input data.

    Options:
      --save-all                 Save intermediate datasets for all tasks in the
                                 pipeline.
      --gpu-id INTEGER           The GPU ID of the device to use.
      --reslice-dir DIRECTORY    Directory for temporary files potentially needed
                                 for reslicing (defaults to output dir)
      --max-cpu-slices INTEGER   Maximum number of slices to use for a block for
                                 CPU-only sections (default: 64)
      --max-memory TEXT          Limit the amount of memory used by the pipeline
                                 to the given memory (supports strings like 3.2G
                                 or bytes)
      --monitor TEXT             Add monitor to the runner (can be given multiple
                                 times). Available monitors: bench, summary
      --monitor-output FILENAME  File to store the monitoring output. Defaults to
                                 '-', which denotes stdout
      --help                     Show this message and exit.

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

- :code:`--save-all`
- :code:`--reslice-dir`
- :code:`--max-cpu-slices`
- :code:`--max-memory`
- :code:`--monitor`
- :code:`--monitor-output`

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

:code:`--max-cpu-slices`
~~~~~~~~~~~~~~~~~~~~~~~~

This flag is only relevant only for runs which are using a pipeline that contains
1 or more sections that are composed of purely CPU methods.

Understanding this flag's usage is dependent on knowledge of the concept of
"chunks", "blocks", and "sections" within HTTomo's framework, so please refer to
:ref:`detailed_about` for information on these concepts.

The notion of a block is fully utilised to increase performance when a sequence of
two or more GPU methods are being executed. When two or more CPU methods are
executed in sequence, the notion of a block plays a less significant role in
performance. The number of slices in a block is driven by the memory capacity of
the GPU, but if no GPU is being used for executing a sequence of methods in the
pipeline, there is no obvious way to choose the number of slices in a block (the
"block size").

In such cases the user may wish to tweak the block size to explore if a specific
block size happens to improve performance for the CPU-only section(s).

:code:`--max-memory`
~~~~~~~~~~~~~~~~~~~~

HTTomo supports running on both:

- a compute cluster, where RAM on the host machine is often quite large
- a personal machine, where RAM is not nearly as large

This is done by a mechanism within HTTomo to hold data in RAM wherever there is
enough RAM to do the required processing, and write data to a file if there is not
enough RAM.

The :code:`--max-memory` flag is for telling HTTomo how much RAM the machine has,
so then it can switch to using a file during execution of the pipeline if
necessary.

:code:`--monitor`
~~~~~~~~~~~~~~~~~

HTTomo has the capability of reporting information about the performance of the
various methods involved in the specific pipeline that will be executed.
Specifically:

- time taken for methods to execute on the CPU/GPU
- transfer time to and from the GPU
- time taken to write to files (if HTTomo uses a file instead of RAM to hold data
  during pipeline execution)

There are two options for this flag, :code:`summary` and :code:`bench`.

:code:`--monitor=summary`
^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`summary` option will produce a brief summary of the time taken for each
method to execute in the pipeline, which will look something like the following:

.. code-block:: console

    Summary Statistics (aggregated across 1 processes):
      Total methods CPU time:     19.376s
      Total methods GPU time:     19.042s
      Total host2device time:      0.013s
      Total device2host time:      0.548s
      Total sources time    :      0.063s
      Total sinks time      :      0.028s
      Other overheads       :      0.362s
      ---------------------------------------
      Total pipeline time   :     19.829s
      Total wall time       :     19.829s
      ---------------------------------------
    Method breakdowns:
                        data_reducer :      0.001s ( 0.0%)
                      find_center_vo :     11.586s (58.4%)
                      remove_outlier :      3.312s (16.7%)
                           normalize :      0.334s ( 1.7%)
         remove_stripe_based_sorting :      2.987s (15.1%)
                                 FBP :      0.966s ( 4.9%)
              save_intermediate_data :      0.019s ( 0.1%)
                      save_to_images :      0.171s ( 0.9%)

:code:`--monitor=bench`
^^^^^^^^^^^^^^^^^^^^^^^

The :code:`bench` option (short for "benchmark") provides a much more in-depth
breakdown of the time taken for each method to execute (dividing it into time
taken on CPU vs. GPU, data transfer times to and from the GPU), and providing this
information for all processes involved in the run.

This output is very verbose, but can provide some insight if, for example, wanting
to see what parts of the pipeline may be slower than expected.

:code:`--monitor-output`
~~~~~~~~~~~~~~~~~~~~~~~~

By default the output of any usage of the :code:`--monitor` flag will be written
to :code:`stdout` (ie, printed to the terminal). However, there are times when
it's useful to write the monitoring output to a file, such as for performance
analysis.

HTTomo supports writing the monitoring results in CSV format, and so any given
filepath to the :code:`--monitor-output` flag will produce a file with the
benchmarking results written in CSV format.
