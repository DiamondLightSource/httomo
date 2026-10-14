.. _faq:

Frequently asked questions
==========================

YAML and pipelines
------------------

.. dropdown:: What is YAML and how does HTTomo use it?

   YAML is a human-readable format for structured configuration files. HTTomo
   uses YAML to describe a :ref:`pipeline
   <explanation_pipelines_templates>` containing the loader and processing
   methods.

   See the :ref:`pipeline_file_reference` for YAML formatting, supported method
   fields, side-output references and parameter-sweep syntax.

.. dropdown:: What is a YAML template?

   A YAML template configures one loader or processing method. It identifies the
   method, its Python module and its configurable parameters.

   Templates can be copied from :ref:`reference_templates` and adapted for a
   particular dataset or processing task.

.. dropdown:: What is a pipeline?

   A pipeline is a YAML file containing an ordered sequence of method
   templates. Older documentation may call it a *process list*.

   The first entry must be a loader. Subsequent methods are executed in order
   from top to bottom, with each method receiving the data produced by the
   preceding operation.

   See :ref:`explanation_pipelines_templates` for an introduction to pipelines
   and templates.

.. dropdown:: How do I build a pipeline?

   Start with a compatible loader template, then add processing templates in
   execution order. Edit the parameters for your data and processing
   requirements.

   The recommended workflow is:

   #. Select methods from :ref:`reference_templates`.
   #. Copy their templates into one YAML file.
   #. Place the loader first.
   #. Configure the method parameters.
   #. Validate the completed pipeline.
   #. Run the pipeline.

   See :ref:`howto_process_list` for detailed instructions and
   :ref:`tutorials_pl_templates` for complete examples.

.. dropdown:: How do I validate a pipeline?

   Use the HTTomo YAML checker before running the pipeline:

   .. code-block:: console

      python -m httomo check pipeline.yaml

   You can optionally provide the input data file so that HTTomo also validates
   its dataset paths:

   .. code-block:: console

      python -m httomo check pipeline.yaml input.nxs

   The checker detects malformed YAML, unknown methods, invalid parameters and
   incompatible dataset paths. See :ref:`utilities_yamlchecker` for details.

.. dropdown:: Can I create a method template?

   Yes. A template can be written manually or generated from a supported backend
   method. It must provide the correct ``method``, ``module_path`` and
   ``parameters`` fields.

   For reusable integration, add the method and its metadata to
   ``httomo-backends``. See :ref:`developers_add_own_method`.

Using and extending HTTomo
--------------------------

.. dropdown:: How do I run HTTomo?

   First install or load HTTomo, prepare a validated pipeline and select the
   input data.

   See :ref:`How to run HTTomo <howto_run>` or
   :ref:`howto_run_at_diamond` when working at Diamond Light Source.

.. dropdown:: Can HTTomo run my own Python method?

   Usually, provided that:

   - the method is importable in the HTTomo environment
   - its execution properties are described in ``httomo-backends``
   - a YAML template is available
   - its inputs and outputs are compatible with an existing
     :ref:`wrapper <info_wrappers>`

   A new wrapper may be required for methods with unusual inputs or outputs. See
   :ref:`developers_add_own_method` for the integration procedure.

.. dropdown:: How can I contribute?

   Contributions can add backend methods, improve documentation or modify the
   `HTTomo source code
   <https://github.com/DiamondLightSource/httomo>`_.

   See :ref:`developers_howtocontribute` for development guidance.

.. dropdown:: Where should I start when a run fails?

   Read ``user.log`` and then ``debug.log`` in the run directory. Validate the
   pipeline against the input file with ``python -m httomo check`` and consult
   :ref:`troubleshooting` for data, MPI, HDF5, CUDA and memory problems.
