Process Lists Guide
********************

In this section we describe how a process list (aka pipeline) can be configured. We
explain how to begin building and editing a process list in general using the
pre-existing templates, and how to :ref:`howto_process_list`.
:ref:`howto_proc_httomo_params` also play a role in defining a process list, so
they are introduced here too.

Editing process lists
---------------------

This section explains how to build a process list (see more on :ref:`explanation_process_list`) from YAML templates
(see more on :ref:`explanation_templates`).

Given time working with HTTomo, a user will likely settle on a workflow for
defining process list YAML files that suits their individual needs. For editing
YAML files, we can recommend Visual Studio Code, Atom, and Notepad++ as editors
that recognise YAML syntax out-of-the-box.

As a starting point, the general process of building the pipeline can be the following:

- copy+paste templates for the desired methods from the
  :ref:`reference_templates` section
- manually edit the parameter values within the copied template as needed. The user might want
  to check the documentation for the relevant method in the library itself.
- intermittently run the :ref:`YAML checker <utilities_yamlchecker>` during
  editing of the YAML file to detect any errors early on. It is strongly recommended to run
  the checker at least once when the YAML pipeline is configured and ready to be run.

Methods order
-------------

Some general rules for building a process list from individual methods are the
following:

* Any process list needs to start with an :ref:`HTTomo loader<reference_loaders>`,
  which are provided as :ref:`reference_templates`.
* The execution order of the methods in the process list is **sequential** starting
  from the top and ending at the bottom.
* The exchange of additional data between methods is performed using
  :ref:`howto_proc_httomo_params`.

.. toctree::
   :maxdepth: 2

   process_lists/httomo_parameters
   process_lists/process_list_configure
