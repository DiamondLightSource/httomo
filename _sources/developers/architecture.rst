.. _developer_architecture:

Architecture and internals
==========================

HTTomo separates orchestration from scientific processing. Backend libraries
implement algorithms, ``httomo-backends`` describes how those functions behave
inside a pipeline, and HTTomo validates, schedules and executes them.

Execution path
--------------

#. The CLI loads and validates the pipeline configuration.
#. The UI and transform layers construct method wrappers and insert framework
   operations such as intermediate saving.
#. The pipeline is divided into sections with compatible processing patterns.
#. The loader supplies data to the runner, which distributes each section into
   per-rank chunks and memory-sized blocks.
#. Wrappers invoke backend methods and pass the resulting block onwards.
#. Dataset stores handle in-memory or disk-backed transitions between sections.

Read :ref:`execution model <fig_execution_model>` and :ref:`Core concepts
<detailed_about>` first. The implementation-oriented concepts are:

.. toctree::
   :maxdepth: 1

   ../introduction/concepts/wrappers
   ../introduction/concepts/memory_estimators

Repository responsibilities
---------------------------

``httomo``
   Pipeline validation, loading, orchestration, MPI distribution, data stores,
   wrappers, monitoring and command-line behaviour.

Backend library
   The scientific implementation and its numerical tests.

``httomo-backends``
   Execution metadata, generated method templates, memory estimation, padding
   and output-shape information. See :ref:`developers_httomo_backends`.

Most new scientific methods require changes to the backend library and
``httomo-backends`` only. Change HTTomo when orchestration or wrapper behaviour
must also change.

Public entry points
-------------------

The :ref:`api` is exhaustive and includes implementation modules. New code
should depend on the narrowest stable interface available, particularly the
pipeline, method-wrapper, loader, method-repository and monitoring interfaces.
Avoid importing from generated documentation paths or relying on private names.
