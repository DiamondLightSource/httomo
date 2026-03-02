.. _explanation_templates:

What is a template?
------------------------

A YAML template (see :ref:`explanation_yaml`) is a textual interface to a method, which can be executed in HTTomo.
The template provides a communication with a chosen method by setting its input/output entries and also additional parameters, if required.

The combination of YAML templates results in a processing list, also called a pipeline. See more on :ref:`explanation_process_list`

As a simple template example, let's consider the template for the median filter from the `TomoPy <https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.corr.html#tomopy.misc.corr.median_filter3d>`_ package.

.. code-block:: yaml

    - method: median_filter3d
      module_path: tomopy.misc.corr
      parameters:
        size: 3

The first two lines are self-explanatory: the method's name and path to the module. HTTomo will interpret this in Python
by importing the method as:

.. code-block:: python

    from tomopy.misc.corr import median_filter3d

The set of parameter values for that method is given in the *parameters* field.

.. note:: Please note that in the original TomoPy's method, there is also the :code:`arr` parameter. This parameter is not exposed in the template because HTTomo will deal with all I/O aspects behind the scenes by using special wrappers. More on that in :ref:`detailed_about`. `YAML generator <https://diamondlightsource.github.io/httomo-backends/utilities/yaml_generator.html>`_ will automatically generate the ready-to-be-used templates.

Please see the list of all supported :ref:`reference_templates`.
