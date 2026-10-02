HTTomo documentation
====================

HTTomo is a Python framework for high-performance tomographic data processing.
It coordinates distributed I/O and CPU/GPU processing through MPI and runs
pipelines described in readable YAML files.

Where should I start?
---------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: New to HTTomo?

      Follow the :ref:`quickstart` to download a small test dataset, validate
      a pipeline and run it from start to finish.

   .. grid-item-card:: Preparing your own data?

      Read :ref:`loading_data`, including how to create an
      :ref:`HTTomo-compatible NXtomo file <create_nxtomo>`.

   .. grid-item-card:: Building a pipeline?

      Browse the :ref:`ready-to-use pipelines
      <tutorials_pl_templates>`, or configure one from
      :ref:`available method templates <reference_templates>`.

   .. grid-item-card:: Developing HTTomo or a method?

      Start with :ref:`developer_architecture`, then follow the contribution
      path that matches your change.

.. _intro_content:

.. toctree::
    :caption: Introduction
    :maxdepth: 2

    introduction/about
    introduction/execution_model
    introduction/data_proc_concepts

.. _getting_started:

.. toctree::
    :caption: Getting started
    :maxdepth: 2

    howto/installation
    getting_started/quickstart
    howto/run_httomo
    getting_started/at_diamond

.. _how_to_content:
.. _tutorials_content:

.. toctree::
    :caption: User guide
    :maxdepth: 2

    howto/loading_data
    howto/httomo_features
    howto/troubleshooting
    howto/tutorial
    faq/faq

.. _pipelines_methods_content:
.. _backends_content:

.. toctree::
    :caption: Pipelines and methods
    :maxdepth: 2

    pipelines/yaml
    pipelines/versioned_downloads    
    pipelines/choose_pipeline
    backends/templates
    backends/list
    pipelines/reconstruction_ecosystem

.. _reference_content:

.. toctree::
    :caption: Reference
    :maxdepth: 2

    reference/cli
    reference/pipeline_file
    reference/run_output
    reference/compatibility
    reference/glossary


.. _developers_content:

.. toctree::
    :caption: Developers
    :maxdepth: 2

    developers/architecture
    developers/development_setup
    developers/how_to_contribute
    developers/add_own_method
    developers/httomo_backends
    developers/memory_calculation
    developers/profiling_tracing
    developers/api
