Process Lists Guide
********************

In this tutorial, we demonstrate how the process lists aka pipelines can be configured. We explain how to perform editing of process lists in general using 
the pre-existing templates and how to :ref:`howto_process_list`. It is also useful to understand what :ref:`howto_proc_httomo_params` are and how they function. 

---------------------
Editing process lists
---------------------

This section explains how to build a process list (see more on :ref:`explanation_process_list`) from YAML templates 
(see more on :ref:`explanation_templates`).

Given time working with HTTomo, a user will likely settle on a workflow for
defining process list YAML files that suits their individual needs. For editing YAML files, we can recommend 
Visual Studio Code, Atom, Notepad++ editors that recognise YAML syntax. 

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

The general rule for building the multitasked process list is the following: 

* Any process list needs to start with :ref:`reference_loaders` which are provided as :ref:`reference_templates`.
* The execution order of the methods in the process list is **sequential** starting from the top to the bottom.
* The exchange of additional data between method is performed using :ref:`howto_proc_httomo_params`.

.. toctree::
   :maxdepth: 2
   
   process_lists/httomo_parameters
   process_lists/process_list_configure   
