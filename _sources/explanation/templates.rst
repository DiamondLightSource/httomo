.. _explanation_templates:

What is a template?
------------------------

A YAML template (see :ref:`explanation_yaml`) in HTTomo is an interface to a method which can be used to build a pipeline of tasks or a process list that will be executed. See more on :ref:`explanation_process_list`
The template provides communication with a method by setting input/output entries and also additional parameters, if required.

Let's consider this simple template for a median filter from the `TomoPy <https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.corr.html#tomopy.misc.corr.median_filter3d>`_ package.

.. code-block:: yaml
    
    tomopy.misc.corr:
      median_filter3d:
        data_in: tomo
        data_out: tomo
        size: 3

The first line :code:`tomopy.misc.corr` specifies the module in TomoPy library and the second :code:`median_filter3d` is the 
name of the method in that module. HTTomo will interpret this in Python
by importing the method:

.. code-block:: python

    from tomopy.misc.corr import median_filter3d

After the method's name follows the list of parameters for that method. Parameters :code:`data_in` 
and :code:`data_out` define the names of the input and the output datasets respectively. Then :code:`size` 
is specific to the method parameter.

Please see the list of all supported :ref:`reference_templates`.
