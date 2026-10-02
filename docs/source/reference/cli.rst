.. _run-httomo-indepth:

Command-line interface
======================

HTTomo provides a command-line interface (CLI) for validating and running
processing pipelines.

Before using the CLI:

* outside Diamond, activate the environment in which HTTomo is installed;
* at Diamond, run ``module load httomo``.

Outside Diamond, invoke the CLI using ``python -m httomo``. Diamond users can
use ``httomo`` as a shortcut.

To list the available commands, run:

.. code-block:: console

   $ python -m httomo --help

The main commands are:

``check``
   Validate a YAML pipeline, optionally against an input HDF5 file.

``memory-check``
   Estimate the peak CPU memory required to run a pipeline.

``run``
   Run a pipeline on an input dataset.

Use ``--help`` after any command to see its current arguments and options:

.. code-block:: console

   $ python -m httomo run --help


The ``check`` command
+++++++++++++++++++++

Validate a YAML pipeline before running it:

.. code-block:: console

   $ python -m httomo check PIPELINE [IN_DATA_FILE]

``PIPELINE``
   Path to the YAML pipeline to validate.

``IN_DATA_FILE``
   Optional path to the input HDF5 file. When supplied, HTTomo also checks that
   the dataset paths referenced by the pipeline loader exist in the file.

For details of the validation performed, see :ref:`utilities_yamlchecker`.

.. note::

   The ``check`` command accepts pipeline files only. Checking a pipeline
   supplied as a string is not currently supported.


The ``memory-check`` command
++++++++++++++++++++++++++++

Estimate the peak CPU memory required to process a dataset:

.. code-block:: console

   $ python -m httomo memory-check IN_DATA_FILE PIPELINE NPROCS

``IN_DATA_FILE``
   Path to the input HDF5 file.

``PIPELINE``
   Path to the pipeline that will process the data.

``NPROCS``
   Number of processes that will run the pipeline. The value must be at least
   one.

The reported value is the estimated peak memory across all processes. It is
calculated from the estimated peak memory for one process multiplied by
``NPROCS``.

The estimate accounts for the input data type and dimensions, loader previews,
padding, re-slicing and changes in data shape between pipeline sections.
See :ref:`memory_and_performance` for planning a process count and comparing
this total with the per-process runtime ceiling.


The ``run`` command
+++++++++++++++++++

Run a processing pipeline:

.. code-block:: console

   $ python -m httomo run [OPTIONS] IN_DATA_FILE PIPELINE OUT_DIR

Arguments
#########

``IN_DATA_FILE``
   Path to an existing HDF5 input file.

``PIPELINE``
   Path to a YAML pipeline. HTTomo can also accept a JSON pipeline supplied as
   a string when ``--pipeline-format json`` is used.

``OUT_DIR``
   Parent directory in which HTTomo creates the run output directory.

By default, the output directory is named using the start time of the run:

.. code-block:: text

   DD-MM-YYYY_HH_MM_SS_output

For example, a run started at 15:30:45 on 1 May 2023 with ``OUT_DIR`` set to
``/home/myuser`` would write to:

.. code-block:: text

   /home/myuser/01-05-2023_15_30_45_output/


Options
#######

Output and intermediate data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--output-folder-name DIRECTORY``
   Use the given output-directory name instead of the timestamp-based default.
   For example, ``--output-folder-name test-1`` creates ``OUT_DIR/test-1``.

``--save-all``
   Save intermediate datasets for every task in the pipeline. Without this
   option, datasets are saved only for tasks whose ``save_result`` setting is
   enabled, either explicitly in the pipeline or by the method's default
   configuration.

``--save-snapshots``
   Save image snapshots at selected points in the pipeline. Snapshots are
   useful for inspecting intermediate processing without saving every complete
   intermediate dataset.

``--intermediate-format hdf5``
   Store intermediate datasets in HDF5 format. This is currently the only
   supported intermediate format and is selected by default.

``--compress-intermediate``
   Store intermediate datasets in chunked HDF5 files with BLOSC compression.

``--frames-per-chunk INTEGER``
   Set the number of frames per HDF5 chunk for intermediate data. The value
   must be at least ``-1``:

   * ``-1`` selects the chunk size automatically and is the default;
   * ``0`` uses contiguous storage;
   * a positive value sets the number of frames per chunk.

   Compression requires chunked storage. If ``--compress-intermediate`` is
   combined with ``--frames-per-chunk 0``, HTTomo changes the chunk setting to
   ``-1`` and selects it automatically.

``--recon-filename-stem NAME``
   Set the filename stem used for reconstruction output. HTTomo adds the
   ``.h5`` extension. For example, ``--recon-filename-stem my-recon`` produces
   ``my-recon.h5``.


Execution and resource use
~~~~~~~~~~~~~~~~~~~~~~~~~~

``--gpu-id INTEGER``
   Select the GPU device to use. The default is ``-1``, which does not
   explicitly select a different CUDA device.

``--max-memory SIZE``
   Set a per-process memory ceiling. Values may be supplied as bytes or with a
   ``K``, ``M`` or ``G`` suffix, for example
   ``--max-memory 32G``.

   When the estimated memory for a pipeline section reaches this limit, HTTomo
   uses disk-backed intermediate storage. For GPU sections, the same value also
   caps the memory budget used to calculate the block size; HTTomo uses the
   smaller of this ceiling and the available GPU memory. The default is ``0``,
   which disables the user-supplied ceiling. GPU block sizing still respects
   the memory reported by the device.

   See :ref:`memory_and_performance` for practical sizing guidance.

``--max-cpu-slices INTEGER``
   Set the maximum number of slices in a block for CPU-only pipeline sections.
   The value must be at least one and defaults to ``64``.

   Adjusting this value may affect the performance of CPU-only processing. See
   :ref:`detailed_about` for information about blocks, chunks and sections.

``--reslice-dir DIRECTORY``
   Choose the directory used for temporary re-slicing files. The directory
   must already exist and be writable. The run output directory is used by
   default.

   When the output is on network-mounted storage, using a local temporary
   directory can substantially improve file-based re-slicing performance. For
   a multi-node run, the directory must be accessible to every participating
   process.

``--continuous-scan-subset START STOP``
   Select a subset of projections along the angular dimension. This option
   overrides the ``continuous_scan_subset`` value in the pipeline loader
   configuration. See :ref:`continuous_scan_subset_selection`.

``--mpi-abort-hook``
   Abort all MPI processes when any process encounters an unhandled exception.
   This prevents the remaining processes from waiting indefinitely for a failed
   process.

   This option is mainly intended for debugging. Because termination occurs at
   the MPI level, the exception traceback may be incomplete.


Pipeline format and parameter sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _pipeline-format:

``--pipeline-format {yaml,json}``
   Select the pipeline format. The value is case-insensitive and defaults to
   YAML.

   YAML pipelines must be provided as files. JSON pipelines must be provided
   as strings.

``--bits-sweep-images INTEGER``
   Set the bit depth of TIFF images produced by a
   :ref:`parameter_sweeping` run. Use ``8``, ``16`` or ``32``. The default is
   ``32``.

   The CLI currently accepts any integer, although the supported output bit
   depths are 8, 16 and 32.


Monitoring
~~~~~~~~~~

``--monitor NAME``
   Enable a performance monitor. The available monitors are ``summary`` and
   ``bench``. This option can be supplied more than once.

   ``summary``
      Report aggregate timings and a per-method breakdown.

   ``bench``
      Report detailed timings for every process, including CPU and GPU
      execution, data transfers and file operations.

``--monitor-output FILENAME``
   Write monitoring results to a file. By default, results are written to
   standard output.

   The ``summary`` monitor produces human-readable text, while the ``bench``
   monitor produces CSV data.


System logging
~~~~~~~~~~~~~~

``--syslog-host HOST``
   Set the hostname of the syslog server. The default is ``localhost``.

``--syslog-port PORT``
   Set the syslog server port. The default is ``514``.
