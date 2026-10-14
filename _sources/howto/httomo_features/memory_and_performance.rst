.. _memory_and_performance:

Memory and performance
======================

HTTomo divides each :term:`section` between MPI processes and then processes
each :term:`chunk` in memory-sized :term:`blocks <block>`. Two controls help
when planning a run: ``memory-check`` estimates host-memory demand before the
run, while ``--max-memory`` limits memory use by each process at runtime.

Estimate CPU memory before a run
--------------------------------

Run the estimate with the same input, pipeline and process count that will be
used for processing:

.. code-block:: console

   $ python -m httomo memory-check INPUT.nxs PIPELINE.yaml NPROCS

The result is an estimated peak across all ``NPROCS`` processes. Divide it by
``NPROCS`` for an approximate per-process value. The calculation includes the
input data type and preview, section padding, shape changes and memory required
for a :term:`re-slice`.

Leave additional capacity for Python, MPI, HDF5, the operating system and other
jobs. The command estimates HTTomo's main section storage; it is not a guarantee
that the complete process will remain below the reported value.

Choose a process count
----------------------

More processes divide the section data into smaller chunks, but every process
has runtime overhead and may allocate method-specific buffers. For GPU runs,
start with one process per GPU. For CPU runs, increase the process count only
while the machine has sufficient memory and I/O bandwidth.

Use a runtime memory ceiling
----------------------------

``--max-memory`` is a **per-process** ceiling, unlike the total reported by
``memory-check``:

.. code-block:: console

   $ python -m httomo run INPUT.nxs PIPELINE.yaml OUTPUT \
       --max-memory 32G

When a section's estimated host-memory requirement reaches the ceiling, HTTomo
uses a temporary HDF5-backed store instead of keeping the section data in RAM.
For GPU sections, the same value caps the memory budget used to choose block
sizes; available device memory still provides an upper bound. A value of ``0``
disables the user ceiling.

Disk-backed sections protect memory at the cost of extra I/O. The warning
``Chunk does not fit in memory - using a file-based store`` indicates that this
path was selected. Put ``--reslice-dir`` on fast storage that every
participating process can access.

Reduce resource use
-------------------

If an estimate or run is too large:

* crop unused detector regions with :term:`preview`;
* process a smaller angular range while developing a pipeline;
* reduce the number of simultaneous processes if aggregate host memory is the
  limit;
* lower ``--max-cpu-slices`` for CPU-only sections;
* lower ``--max-memory`` to reduce GPU blocks or select disk-backed storage;
* avoid ``--save-all`` unless every intermediate result is needed; and
* use ``--compress-intermediate`` when storage capacity matters more than
  compression overhead.

For the complete option definitions, see :ref:`run-httomo-indepth`. Developers
implementing or correcting method estimates should read
:ref:`developers_memorycalc`.
