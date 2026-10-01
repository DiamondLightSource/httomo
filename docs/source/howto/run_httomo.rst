.. _howto_run:

Run your first pipeline
=======================

This guide takes you through choosing, checking and running an existing HTTomo
pipeline. If HTTomo is not installed yet, follow :ref:`installation_main`
first.

Required inputs
---------------

You need:

* an HDF5 file containing the input tomography data;
* a YAML pipeline, also called a *process list*, describing the processing;
* a directory in which HTTomo can create its output.

The input data should normally follow the `NXtomo application definition
<https://manual.nexusformat.org/classes/applications/NXtomo.html>`_. HTTomo's
standard loader can automatically locate the projections, image keys and
rotation angles in an NXtomo file. Other HDF5 layouts can be used by setting
the dataset paths explicitly in the loader configuration.

If you need sample data, the :ref:`synthetic-data-example` page explains how to
generate an NXtomo file. :ref:`Real data <real-data-example>` can also be used directly.

Choose an example pipeline
--------------------------

Start with one of the :ref:`ready-to-use pipelines
<tutorials_pl_templates>`. You can also find data and the associated pipelines in :ref:`data_tutorials`.
Choose a GPU pipeline when a CUDA-enabled GPU and its required processing libraries are available. Otherwise, choose a CPU
pipeline and ensure that its backend library, such as TomoPy, is installed.

Copy the selected pipeline into a local YAML file. Check its loader entry and
adapt the data paths if the input is not an NXtomo file. For guidance on
changing methods or parameters, see :ref:`how_to_configure_pipeline`.

.. admonition:: HTTomo at Diamond Light Source
   :class: note

   If you are running HTTomo at Diamond, see the dedicated page on
   :ref:`howto_run_at_diamond`.

Validate the pipeline
---------------------

Before running the pipeline, check that the process list is valid and that its
data paths exist in the input file:

.. code-block:: console

   $ python -m httomo check PIPELINE.yaml INPUT.h5

Replace ``PIPELINE.yaml`` and ``INPUT.h5`` with the paths to the chosen
pipeline and input data. A successful check confirms that the YAML structure,
methods, parameters and referenced HDF5 paths are valid. Correct any reported
errors before continuing. See :ref:`utilities_yamlchecker` for details of the
checks performed.

Run HTTomo
----------

Then run the pipeline:

.. code-block:: console

   $ python -m httomo run INPUT.h5 PIPELINE.yaml OUTPUT_DIR

Replace the example arguments as follows:

``INPUT.h5``
   The path to the HDF5 file containing the input data.

``PIPELINE.yaml``
   The path to the YAML process list that defines the processing pipeline.

``OUTPUT_DIR``
   The directory in which HTTomo will create the processing output.

HTTomo creates a timestamped run directory inside ``OUTPUT_DIR``. For all
available commands and options, see :ref:`run-httomo-indepth`.

Run in parallel
---------------

To divide the input data between multiple processes, launch HTTomo using MPI:

.. code-block:: console

   $ mpirun -np N python -m httomo run INPUT.h5 PIPELINE.yaml OUTPUT_DIR

Here, ``N`` is the number of parallel processes to launch. Each process
normally requires access to a GPU when the pipeline contains GPU methods.

At Diamond, use the ``httomo_mpi`` launcher instead; see
:ref:`howto_run_at_diamond`.

Find and inspect the output
---------------------------

Open the newly created run directory inside ``OUTPUT_DIR``. It contains:

* ``pipeline.yaml``, a copy of the pipeline used for the run;
* ``user.log``, containing the concise progress information shown in the
  terminal;
* ``debug.log``, containing more detailed diagnostic information;
* requested intermediate or reconstruction results as HDF5 files; and
* snapshots or image files when the corresponding output options or pipeline
  methods were enabled.

Inspect HDF5 results with an HDF5-compatible viewer such as DAWN, HDFView or
silx. See :ref:`info_logger` for help interpreting the logs, progress
bars and other output files.


.. toctree::
   :maxdepth: 2

   how_to_run/at_diamond
