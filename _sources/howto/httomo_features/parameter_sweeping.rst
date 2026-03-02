.. _parameter_sweeping:

Parameter Sweeping
^^^^^^^^^^^^^^^^^^

What is it?
===========

Parameter sweeping refers to providing multiple values for a specific parameter
of a method, and then running that method on its input data with the different
values for that parameter.

How would this be useful when processing data?
==============================================

This feature is typically used when prototyping a process list and it is
difficult to guess a reasonable value of a method's parameter. There could be
many reasons for this situation, such as being unfamiliar with the method, or
working with unfamiliar data, etc.

How the output looks like?
==========================

.. _fig_centergif:
.. figure::  ../../_static/sweep/sweep_cor.gif
    :scale: 55 %
    :alt: Sweep for cor

    The result of sweeping applied to find the correct :ref:`centering`. The images with the CoR values printed on them are saved in the folder. The user can find the correct value by looking through the images and then input the value manually into the pipeline, see :ref:`centering_manual`.


How are parameter sweeps defined in the process list YAML file?
===============================================================

There are two ways of specifying the values that a parameter sweep should be
performed across:

1. Specifying a range of values via start, stop and step values

2. Manually specifying each value

.. note:: A pipeline can only have 1 parameter sweep in it at a time. Any
   pipelines with more than 1 parameter sweep defined in it will not be
   executed, and an error message will be displayed.

Specifying a range
++++++++++++++++++

The first way is done by providing a start, stop and step value. Along with this
information, a special phrase :code:`!SweepRange` is used to "mark" in the YAML
that the start, stop and step values are for defining a parameter sweep.

The snippet below is defining a parameter sweep for the :code:`center` parameter
of a reconstruction method, where the sweep starts at :code:`10`, ends at
:code:`40` (similar to python slicing, the end value is not included), with steps
of :code:`10` inbetween:

.. code-block:: yaml

    center: !SweepRange
      start: 10
      stop: 50
      step: 10

Specifying each value
+++++++++++++++++++++

The second way is done by providing a list of values for a parameter, and again
"marking" the list with a special phrase to denote that this list of values is
defining a parameter sweep. The phrase in this case is :code:`!Sweep`.

The snippet below is defining a parameter sweep for the :code:`size` parameter
of a median filter method, where the sweep is across the two values :code:`3`
and :code:`5`:

.. code-block:: yaml

    size: !Sweep
      - 3
      - 5

Example
+++++++

Below, :code:`!Sweep` is used in the context of a fully working minimal
pipeline to sweep over the :code:`size` parameter of a median filter:

.. literalinclude:: ../../../../tests/samples/pipeline_template_examples/testing/sweep_manual.yaml
  :language: yaml

.. note:: There is no need to add image saving method after the `sweep` method. The result of the `sweep` method will be saved into images automatically. 

How big should the input data be?
=================================

Due to the goal of parameter sweeps being to provide quick feedback to optimise
a parameter value, it is typical to run a parameter sweep on a few sinograms,
rather than the full data.

As such, a parameter sweep run in HTTomo is constrained to run on data previewed
to contain 7 sinogram slices or less. Meaning, in order to perform a parameter
sweep in a pipeline, the input data must be cropped to 7 sinogram slices or less
using the :code:`preview` parameter of the loader (see :ref:`previewing` for
more details), otherwise the parameter sweep run will not execute and an error
message will be displayed.

What structure does the output data of a parameter sweep have?
==============================================================

When a parameter sweep is executed, the output of the method will be the set of
middle slices from each individual result of the sweep (sinogram slices or recon
slices), collected along the middle dimension.

For example, suppose:

- the input data is previewed to 3 sinogram slices and has shape
  :code:`(1801, 3, 2560)`

- a parameter sweep is performed on the :code:`center` parameter of a
  reconstruction method in the pipeline, across 10 different CoR values

In this case, each execution of the reconstruction method will produce 3 slices,
as an array of shape :code:`(2560, 3, 2560)`. So, 10 arrays of shape
:code:`(2560, 3, 2560)` will be produced.

The middle slice from each array of 3 slices will be taken, resulting in 10
reconstructed slices altogether. These 10 reconstructed slices will then be
concatenated along the middle dimension and put into a separate array, resulting
in the final data shape of :code:`(2560, 10, 2560)`.

This output containing 10 slices will then be passed onto the next method in the
pipeline; for example, a method to save the 10 slices as images for quick
inspection.
