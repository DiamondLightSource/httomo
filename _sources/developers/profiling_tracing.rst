.. _profiling_tracing:

Profiling and tracing
*********************

`VizTracer`_ can record CPU traces and CPU and memory usage statistics for
investigating HTTomo performance.

Record a serial run
===================

.. code-block:: console

   $ python -m viztracer \
       --plugin "vizplugins --cpu_usage --memory_usage" \
       --output_file output.json \
       -m httomo run --no-standalone \
       data.nxs pipeline.yaml output_directory

Record an MPI run
=================

Create a separate trace for each MPI rank:

.. code-block:: console

   $ mpirun -n 4 bash -c \
       'python -m viztracer \
       --plugin "vizplugins --cpu_usage --memory_usage" \
       --output_file output_rank_${OMPI_COMM_WORLD_RANK}.json \
       -m httomo run --no-standalone \
       data.nxs pipeline.yaml output_directory'

Combine and inspect traces
==========================

Combine the rank-specific traces:

.. code-block:: console

   $ viztracer --combine output_rank_*.json -o output.json

View the result locally:

.. code-block:: console

   $ vizviewer output.json

VizTracer output can also be opened using the online `Perfetto`_ viewer.

.. _VizTracer: https://viztracer.readthedocs.io/
.. _Perfetto: https://ui.perfetto.dev/
