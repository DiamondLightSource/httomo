
What is a template?
------------------------

A YAML template in HTTomo is an interface to a method which can be used to build a pipeline of tasks that will be run.
Template provides a communication with a method by setting input/output and also additional parameters if required. 

Lets consider this simple template for median filter from `TomoPy <https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.corr.html#tomopy.misc.corr.median_filter3d>`_ package. 

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

