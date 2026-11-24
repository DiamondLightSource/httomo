.. _parameter_sweeping:

Parameter Sweeping
^^^^^^^^^^^^^^^^^^

A :term:`parameter sweep` provides multiple values for one parameter and runs
the method once for each value.

How would this be useful when processing data?
==============================================

Use a parameter sweep when prototyping a pipeline to compare possible
values, especially for an unfamiliar method or dataset.

What does the output look like?
===============================

.. _fig_centergif:
.. figure::  ../../_static/sweep/sweep_cor.gif
    :scale: 55 %
    :alt: Sweep for cor

    A sweep used to find the correct :ref:`centering`. Each saved image shows
    its CoR value. Inspect the images, then enter the best value in the pipeline;
    see :ref:`centering_manual`.


How are parameter sweeps defined in the pipeline YAML file?
===========================================================

Specify sweep values in either of two ways:

1. A range defined by start, stop, and step values.
2. An explicit list of values.

.. note:: A pipeline can contain only one parameter sweep. HTTomo reports an
   error and does not run a pipeline containing more than one.

Specifying a range
++++++++++++++++++

Use :code:`!SweepRange` with :code:`start`, :code:`stop`, and :code:`step`.
Like a Python range, it excludes the stop value. This example sweeps the
reconstruction :code:`center` parameter from 10 to 40 in steps of 10:

.. code-block:: yaml

    center: !SweepRange
      start: 10
      stop: 50
      step: 10

Specifying each value
+++++++++++++++++++++

Use :code:`!Sweep` to provide explicit values. This example sweeps a median
filter's :code:`size` parameter over :code:`3` and :code:`5`:

.. code-block:: yaml

    size: !Sweep
      - 3
      - 5

Example
+++++++

This minimal pipeline uses :code:`!Sweep` for a median filter's :code:`size`
parameter:

.. literalinclude:: ../../../../tests/samples/pipeline_template_examples/testing/sweep_manual.yaml
  :language: yaml

.. note:: Sweep results are saved as images automatically; no image-saving
   method is required.

How big should the input data be?
=================================

Sweeps are intended to give quick feedback on a small input. Use the loader's
:code:`preview` parameter to select no more than seven sinogram slices; see
:ref:`previewing`. HTTomo reports an error and stops if the preview contains
more slices.

What structure does the output data of a parameter sweep have?
==============================================================

The sweep output contains the middle slice from each run, concatenated along
the middle dimension. This applies to both sinogram and reconstruction slices.

For example, suppose:

- the input contains three sinogram slices with shape
  :code:`(1801, 3, 2560)`; and

- a parameter sweep is performed on the :code:`center` parameter of a
  reconstruction method using ten CoR values.

Each reconstruction produces an array with shape :code:`(2560, 3, 2560)`.
HTTomo takes the middle slice from each of the ten arrays and concatenates them
into an output with shape :code:`(2560, 10, 2560)`. It passes this output to the
next pipeline method, such as one that saves the slices for inspection.
