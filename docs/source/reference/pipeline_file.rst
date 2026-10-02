.. _pipeline_file_reference:
.. _explanation_yaml:

Pipeline file reference
=======================

An HTTomo pipeline file is a YAML sequence of method entries. HTTomo executes
the entries from top to bottom, passing the main dataset from one method to the
next. The first entry must be a loader.

YAML formatting
---------------

YAML uses indentation to define structure. Use spaces rather than tabs, keep
indentation consistent and include a space after each colon. Comments begin
with ``#``.

Common values include:

``null``
   No value. This is converted to Python ``None``.

``true`` and ``false``
   Boolean values all in small letters. These are converted to Python ``True`` and ``False``.

``[value1, value2]``
   A list written on one line. Lists may also be written across multiple lines.

Strings normally do not need quotation marks. Quote a string when it contains
YAML punctuation or might otherwise be interpreted as a number, Boolean or
``null`` value.

Method entries
--------------

Each list entry configures one loader or processing method:

.. code-block:: yaml

   - method: median_filter3d
     module_path: tomopy.misc.corr
     parameters:
       size: 3

The supported fields are:

``method``
   Required. The name of the Python function to run.

``module_path``
   Required. The import path containing the function.

``parameters``
   Required. A mapping of parameter names to values. Use ``parameters: {}``
   when the method has no configurable parameters.

``id``
   Optional. A unique name used when another method references this entry's
   side output.

``side_outputs``
   Optional. Maps an output name supplied by the method to a pipeline name. See
   :ref:`side_output`.

``save_result``
   Optional Boolean. Controls whether the main dataset is saved after this
   method. See :ref:`save-result-examples`.

Use :ref:`reference_templates` for the supported method names, module paths,
parameters and defaults.

Side-output references
----------------------

A later method can use a named side output with the following syntax:

.. code-block:: yaml

   ${{method_id.side_outputs.output_name}}

The producing method must appear earlier in the pipeline, and every explicit
``id`` must be unique. See :ref:`side_output` for a complete example.

Parameter sweeps
----------------

Use ``!Sweep`` for an explicit list of parameter values and ``!SweepRange`` for
a start, stop and step. A pipeline can contain only one sweep. See
:ref:`parameter_sweeping` for syntax and output behaviour.

Minimal pipeline
----------------

This example loads an NXtomo file, normalises the projections and applies the
negative logarithm:

.. code-block:: yaml

   - method: standard_tomo
     module_path: httomo.data.hdf.loaders
     parameters:
       data_path: auto
       image_key_path: auto
       rotation_angles: auto

   - method: normalize
     module_path: tomopy.prep.normalize
     parameters:
       cutoff: null
       averaging: mean

   - method: minus_log
     module_path: tomopy.prep.normalize
     parameters: {}

Before running a pipeline, follow :ref:`utilities_yamlchecker` to validate its
structure, methods, parameters and input-data paths.
