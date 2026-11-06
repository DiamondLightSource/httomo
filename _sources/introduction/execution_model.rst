How HTTomo runs a pipeline
==========================

HTTomo reads the YAML pipeline, validates its methods and parameters, and
constructs an executable pipeline. Method wrappers adapt backend functions to
HTTomo's common data-processing interface.

.. _fig_execution_model:

.. figure:: ../_static/execution_model.svg
   :width: 100%
   :alt: Diagram of the HTTomo pipeline execution model

   HTTomo execution from a YAML pipeline to output. The pipeline is divided
   into sections; within each section, data is distributed into chunks and
   processed block by block.

Creating sections
~~~~~~~~~~~~~~~~~

HTTomo groups compatible methods into :ref:`sections <info_sections>`. For each
section, it calculates a safe block size from the available memory and the
requirements of its methods.

Executing a section
~~~~~~~~~~~~~~~~~~~

Within each section, the dataset is divided among the available MPI processes.
Each process receives one :ref:`chunk <chunks_data>` and normally operates
independently on its assigned data.

Each chunk is then divided into :ref:`blocks <blocks_data>` using the block size
calculated for the section. A block passes through every method in the section
before the next block is processed.

Moving between sections
~~~~~~~~~~~~~~~~~~~~~~~

At a section boundary, HTTomo may synchronise the MPI processes, save
intermediate data or redistribute the dataset. If the data pattern changes
between projections and sinograms, HTTomo performs a
:ref:`re-slice <info_reslice>`.

Completing the pipeline
~~~~~~~~~~~~~~~~~~~~~~~

Processing continues section by section until every block has passed through
the pipeline. HTTomo then writes the requested results and monitoring
information.
