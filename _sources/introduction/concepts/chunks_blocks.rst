.. _chunks_blocks_data:

Chunks and blocks
=================

HTTomo divides data at two levels. The dataset is first distributed across MPI
processes as *chunks*. Each chunk is then divided into memory-sized *blocks* for
processing.

.. _chunks_data:

Chunks
------

A *chunk* is the part of the dataset assigned to one MPI process. Distributing
chunks allows multiple processes to work on the data in parallel.


.. admonition:: on "chunk" terminology
   :class: note

   *chunk* is purely an HTTomo term and is unrelated to HDF5 chunks.


.. _fig_chunks:

.. figure:: ../../_static/blocks_chunks/chunks.png
   :alt: Tomographic data distributed as chunks between two MPI processes
   :align: center
   :width: 90%

   Data with shape :code:`(180, 128, 160)` distributed between two MPI
   processes. Each process receives a chunk of shape
   :code:`(90, 128, 160)`.

Chunk size
~~~~~~~~~~

Chunk size depends on the full dataset shape, its current projection or sinogram
orientation, and the number of MPI processes.

HTTomo divides the data as evenly as possible. If an equal division is not
possible, the highest-rank MPI process receives the differently sized chunk.

.. _blocks_data:

Blocks
------

A *block* is a smaller piece of a :ref:`chunk <chunks_data>` or equal to the size of chunk. HTTomo processes
blocks individually so that the data fits within the available CPU or GPU memory.

Block size
~~~~~~~~~~

HTTomo calculates block size at runtime using:

- the available memory
- the memory requirements of every method in the current
  :ref:`section <info_sections>`

Block size may change between sections. If sufficient memory is available, a
single block can contain the entire chunk.

.. _fig_blocks:

.. figure:: ../../_static/blocks_chunks/blocks.png
   :alt: An HTTomo chunk divided into smaller blocks
   :align: center
   :width: 90%

   A chunk of shape :code:`(90, 128, 160)` divided into two blocks of shape
   :code:`(45, 128, 160)`. Each block contains 45 projections and is processed
   individually.

Processing blocks
~~~~~~~~~~~~~~~~~

Blocks are HTTomo's main processing unit. Loaders produce blocks, methods process
them and the resulting blocks continue through the pipeline.

When a chunk contains multiple blocks, they are processed sequentially.

.. dropdown:: More details about block processing

   **Notes on the framework's approach to data**

   HTTomo's framework has been written with GPUs in mind. More specifically,
   HTTomo aims to use as much available GPU memory as possible while remaining
   within a safe limit.

   Even after data is divided into chunks, a chunk may not fit into GPU memory.
   Similarly, the full dataset may not fit into CPU memory. This can occur on both
   compute clusters and personal machines.

   **Why split a chunk into smaller pieces?**

   Each MPI process works with one chunk and is typically associated with one GPU.
   HTTomo cannot assume that the entire chunk will fit into that GPU's memory.

   For example, dividing a 20 GB dataset among four MPI processes produces chunks
   of approximately 5 GB. A cluster GPU may have enough memory for an entire chunk,
   while a personal GPU with 4 GB of memory would not.

   HTTomo therefore divides each chunk into smaller pieces called *blocks*.

   **How are block shapes calculated?**

   HTTomo calculates block sizes during pipeline execution. Blocks within a
   sequence of methods have approximately the same shape, but their shape may
   change between pipeline stages.

   The calculation uses information from the :ref:`library files <pl_library>` and
   is mainly based on:

   - the GPU memory available to the process
   - the memory requirements of the methods in the current
     :ref:`section <info_sections>`

   Block size is expressed as a number of slices. For projection data, each process
   owns a chunk of projections and divides it into blocks containing a suitable
   number of projection slices. The same principle applies to sinogram data.

   **Blocks as the fundamental data quantity**

   Blocks are HTTomo's main processing unit:

   #. Loaders produce individual blocks.
   #. Methods receive individual blocks as input.
   #. Methods produce individual blocks as output.

   If an entire chunk fits into memory, a single block may span the whole chunk.
   HTTomo still treats it as a block.