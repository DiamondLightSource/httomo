.. _faq:

Frequently asked questions
==========================

.. _explanation_yaml:

YAML and pipelines
------------------

.. dropdown:: What is YAML and how does HTTomo use it?

   YAML is a human-readable format for structured configuration files. HTTomo
   uses YAML to describe a :ref:`process list
   <explanation_pipelines_templates>` containing the loader and processing
   methods that form a pipeline.

   YAML uses key-value pairs, lists and indentation:

   .. code-block:: yaml

      # A list containing one method
      - method: median_filter3d
        module_path: tomopy.misc.corr
        parameters:
          size: 3

   In this example:

   - the leading ``-`` starts a list entry
   - ``method``, ``module_path`` and ``parameters`` are keys
   - indentation places ``size`` inside ``parameters``
   - text following ``#`` is a comment

   Important formatting rules:

   - Use spaces for indentation, never tabs.
   - Keep indentation consistent.
   - Include a space after each colon.
   - Use ``true``, ``false`` and ``null`` for Boolean and empty values.
   - Quote strings when they contain special characters or could be
     interpreted as another data type.

.. dropdown:: What is a YAML template?

   A YAML template configures one loader or processing method. It identifies the
   method, its Python module and its configurable parameters.

   Templates can be copied from :ref:`reference_templates` and adapted for a
   particular dataset or processing task.

.. dropdown:: What is a process list?

   A process list is a YAML file containing an ordered sequence of method
   templates. HTTomo interprets this sequence as a processing pipeline.

   The first entry must be a loader. Subsequent methods are executed in order
   from top to bottom, with each method receiving the data produced by the
   preceding operation.

   See :ref:`explanation_pipelines_templates` for an introduction to templates,
   process lists and pipelines.

.. dropdown:: How do I build a pipeline?

   Start with a compatible loader template, then add processing templates in
   execution order. Edit the parameters for your data and processing
   requirements.

   The recommended workflow is:

   #. Select methods from :ref:`reference_templates`.
   #. Copy their templates into one YAML file.
   #. Place the loader first.
   #. Configure the method parameters.
   #. Validate the completed process list.
   #. Run the pipeline.

   See :ref:`howto_process_list` for detailed instructions and
   :ref:`tutorials_pl_templates` for complete examples.

.. dropdown:: How do I validate a process list?

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

   First install or load HTTomo, prepare a validated process list and select the
   input data.

   See :ref:`howto_run_at_diamond` when working at Diamond Light Source, or
   :ref:`howto_run_outside_diamond` for other systems.

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

.. _faq_workstation:

Working at Diamond Light Source
-------------------------------

.. _terminal:

.. dropdown:: What is a terminal?

   A terminal—also called a shell, console or command line—is a text-based
   interface for running commands. HTTomo is loaded, configured and started from
   a terminal.

.. dropdown:: How do I load HTTomo at Diamond?

   List the installed versions with:

   .. code-block:: console

      module avail httomo

   Load the default recommended version with:

   .. code-block:: console

      module load httomo

   This configures the Python environment and makes HTTomo and its dependencies
   available in the current terminal.

.. dropdown:: What does ``module load`` do?

   The module system updates environment variables such as executable and
   library paths. Loading HTTomo activates the configured software environment
   without requiring a separate installation.

   See the `Environment Modules documentation
   <https://modules.readthedocs.io>`_ for more information.

.. dropdown:: How do I change the loaded HTTomo version?

   Unload the current version and then load the required one:

   .. code-block:: console

      module unload httomo
      module load httomo/<version>

   Run ``module avail httomo`` to see the available version names.