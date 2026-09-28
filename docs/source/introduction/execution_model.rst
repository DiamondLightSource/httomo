How HTTomo runs a pipeline
==========================

From process list to pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HTTomo reads the YAML process list, validates its methods and parameters, and
constructs an executable pipeline. Method wrappers adapt backend functions to
HTTomo's common data-processing interface.

Distributing the data
~~~~~~~~~~~~~~~~~~~~~

The dataset is divided among the available MPI processes. Each process receives
one :ref:`chunk <chunks_data>` and normally operates independently on its assigned
data.

Creating sections and blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HTTomo groups compatible methods into :ref:`sections <info_sections>`. For each
section, it calculates a safe block size from the available memory and the
requirements of its methods.

Each chunk is then divided into :ref:`blocks <blocks_data>`. A block passes through
every method in the section before the next block is processed.

Moving between sections
~~~~~~~~~~~~~~~~~~~~~~~

At a section boundary, HTTomo may synchronise the MPI processes, save intermediate
data or redistribute the dataset. If the data pattern changes between projections
and sinograms, HTTomo performs a :ref:`re-slice <info_reslice>`.

Completing the pipeline
~~~~~~~~~~~~~~~~~~~~~~~

Processing continues section by section until every block has passed through the
pipeline. HTTomo then writes the requested results and monitoring information.