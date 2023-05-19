.. _howto_process_list:

Configure process list using templates
======================================

This section explains how to build a process list (see more on :ref:`explanation_process_list`) from YAML templates (see more on :ref:`explanation_templates`).

We focus on several important aspects which can be helpful to keep in mind, before configuring the process list. They are :ref:`pl_conf_order`, :ref:`pl_reslice`, and :ref:`pl_platform_sections`. First two topics are 
recommended to read if you are new to HTTomo and :ref:`pl_platform_sections` is for more in-depth information about the inner workings of HTTomo.

**The better understanding of those elements will enable you to build more computationally efficient pipelines**. 

Before start, please become familiar with :ref:`explanation_yaml` and use editors that support it. We can recommend Visual Studio Code, Atom, Notepad++. 
Also to avoid less comprehensive errors during the run with the new process list, please validate it first using :ref:`utilities_yamlchecker` tool.

.. _pl_conf_order:

Methods order
-------------
To build a process list you will need to copy-paste the content of YAML files from the provided :ref:`reference_templates`.
The general rules for building a process list are the following: 

* Any process list starts with an :ref:`reference_loaders` which is provided as a template (see :ref:`reference_templates`).
* The execution order of the methods in the process list is **sequential** starting from the top to the bottom.

For example, for tomographic processing we can build the following process list by using TomoPy templates. 
and HTTomo loader to read `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ dataset.

.. dropdown:: A basic TomoPy full data processing pipeline

    .. literalinclude:: ../../../samples/pipeline_template_examples/01_basic_cpu_pipeline_tomo_standard.yaml

In this process list the data will be loaded, then normalised, then the centre of rotation will be estimated 
and provided to the reconstruction, and finally the result of the reconstruction will be saved as tiff files. 
Note that the result of the reconstruction will be also saved as an HDF5 file. 

.. _pl_reslice:

Reslicing
-------------


.. _pl_platform_sections:

Platform Sections
-----------------