.. _howto_process_list:

Configure efficient pipelines
=============================

Here we focus on several important aspects which can be helpful to keep in mind while configuring the process list. They are :ref:`pl_conf_order`, :ref:`info_reslice`, and :ref:`info_sections`. The first two topics are
recommended to read if you are new to HTTomo and :ref:`info_sections` is for more in-depth information about the inner workings of HTTomo when using GPUs.

**The better understanding of those elements will enable you to build more computationally efficient pipelines**. 

Please become familiar with :ref:`explanation_yaml` and use editors that support it. We can recommend Visual Studio Code, Atom, Notepad++. 
To avoid errors during the HTTomo run using a process list, please validate it first using YAML checker tool (see :ref:`utilities_yamlchecker`).

.. _pl_conf_order:

Methods order
-------------
To build a process list with multiple tasks, you will need to copy-paste the content of YAML files from the provided :ref:`reference_templates`.
The general rules for building a process list are the following: 

* Any process list starts with :ref:`reference_loaders` which are provided as :ref:`reference_templates`.
* The execution order of the methods in the process list is **sequential** starting from the top to the bottom.

For example, for tomographic processing, we can build the following process list by using mostly TomoPy templates.

.. dropdown:: A basic TomoPy full data processing pipeline

    .. literalinclude:: ../../../../tests/samples/pipeline_template_examples/pipeline_cpu1.yaml

In this process list the data will be loaded using the standard HTTomo loader to read `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ dataset. 
Then the loaded data is normalised, the centre of rotation estimated and provided to the reconstruction. 
Finally the result of the reconstruction will be saved as tiff files using HTTomolib library as a backend. 
Note that the result of the reconstruction will be also saved as an HDF5 file. 

