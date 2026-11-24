.. _optimise_pipeline:

Optimise pipeline performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pipeline performance is influenced mainly by method order, CPU/GPU data
transfers and intermediate file output. Understanding :ref:`info_sections` and
:ref:`info_reslice` can help when applying the guidance below.

.. _pl_conf_order:

Group methods by data pattern
=============================

HTTomo runs pipeline methods sequentially from top to bottom. Each method has
one of three data patterns:

``projection``
   Data is sliced by projection.

``sinogram``
   Data is sliced by sinogram.

``all``
   The method inherits the pattern of the preceding method.

Changing between ``projection`` and ``sinogram`` requires a potentially costly
:ref:`re-slice <info_reslice>`. Where processing requirements allow, group
methods with the same pattern to reduce the number of re-slices.

HTTomo loaders use the ``projection`` pattern, so start with projection-based
methods where possible. Centre-finding methods should normally be placed near
the start of the pipeline.

.. _pl_library:

Method metadata
===============

HTTomo obtains each method's pattern, implementation type and memory
requirements from :ref:`httomo-backends <developers_httomo_backends>`. See the
`httomo-backends method metadata documentation
<https://diamondlightsource.github.io/httomo-backends/backends/method_info.html>`_
for details.

Group GPU methods
=================

Supported methods use one of three implementation types:

``cpu``
   Runs on the CPU.

``gpu``
   Runs on a GPU but receives its input as a NumPy array in CPU memory.

``gpu_cupy``
   Runs on a GPU using CuPy arrays. Data remains in GPU memory between
   consecutive ``gpu_cupy`` methods.

When a GPU is available, prefer GPU implementations where appropriate and keep
``gpu_cupy`` methods together to reduce transfers between CPU and GPU memory.
Available implementations are listed in :ref:`backends_list`.

Minimise writing to disk
========================

Writing intermediate datasets can significantly slow a pipeline and consume
substantial disk space. Use ``save_result`` or ``--save-all`` only when those
intermediate results are needed. Use ``--save-snapshots`` to capture lightweight
diagnostic snapshots instead. See :ref:`save-result-examples` for details.
