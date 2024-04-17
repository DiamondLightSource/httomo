.. _howto_edit_list:

=====================
Editing process lists
=====================

This section explains how to build a process list (see more on :ref:`explanation_process_list`) from YAML templates 
(see more on :ref:`explanation_templates`).

Given time working with HTTomo, a user will likely settle on a workflow for
defining process list YAML files that suits their individual needs.

As a starting point, something like the following is suggested:

- copy+paste templates for the desired methods from the
  :ref:`reference_templates` section
- manually edit the parameter values within the copied template as needed. The user might want 
  to check the documentation for the relevant method in the library itself.
- intermittently run the :ref:`YAML checker <utilities_yamlchecker>` during
  editing of the YAML file to detect any errors early on. It is strongly recommended to run 
  the checker at least once when the YAML pipeline is configured.

Method Parameters vs. HTTomo Method Parameters
----------------------------------------------

In addition to parameters for a method that influence the processing that the
method performs, there are some parameters that can be used in a method's YAML
configuration that modify in some way how HTTomo handles the method (for
example, whether or not to write the output of the method to a file).

Since these parameters are *not* parameters for the method itself and are
instead specific to HTTomo, the next section will go through these extra
parameters, explaining what they are for and giving examples of how they can be
used.

HTTomo-specific Method Parameters
---------------------------------

In the following subsections, we introduce parameters that are **HTTomo-specific** and do NOT belong to the list of the exposed method's parameters from the backend library. 

.. toctree::
   :maxdepth: 1
   
   side_outputs/side_out
   save_results/save_results


