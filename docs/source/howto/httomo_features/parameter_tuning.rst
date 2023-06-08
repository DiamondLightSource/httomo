.. _parameter_tuning:

Parameter Tuning
^^^^^^^^^^^^^^^^

What is it?
===========

Parameter tuning (also called "parameter sweeping") refers to providing multiple
values for a specific parameter of a method, and then running that method on its
input data with the different values for that parameter.

How would this be useful when processing data?
==============================================

This feature is typically used when prototyping a process list and it is
difficult to guess a reasonable value of a method's parameter. There could be
many reasons for this situation, such as being unfamiliar with the method, or
working with unfamiliar data, etc.

.. _parameter_tuning_range:

How are parameter sweeps defined in the process list YAML file?
===============================================================

There are two ways of specifying the values that a parameter sweep should be
performed across:

1. Specifying a range of values via start, stop and step values

2. Manually specifying each value

Specifying a range
++++++++++++++++++

The first way is done by providing a start, stop and step value. Along with this
information, a special phrase :code:`!SweepRange` is used to "mark" in the YAML
that the start, stop and step values are for defining a parameter sweep.

The snippet below is defining a parameter sweep for the :code:`center` parameter
of a reconstruction method, where the sweep starts at :code:`10`, ends at
:code:`50`, with steps of :code:`10` inbetween:

.. code-block:: yaml

    center: !SweepRange
      start: 10
      stop: 50
      step: 10

Below, the same parameter is used in the context of a fully working yet minimal
pipeline taken from the HTTomo repo:

.. literalinclude:: ../../../../samples/pipeline_template_examples/parameter_sweeps/01_recon_cor_range_sweep.yaml
  :language: yaml
  :caption: samples/pipeline_template_examples/parameter_sweeps/01_recon_cor_range_sweep.yaml

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

Below, the same parameter is used in the context of a fully working minimal
pipeline taken from the HTTomo repo:

.. literalinclude:: ../../../../samples/pipeline_template_examples/parameter_sweeps/02_median_filter_kernel_sweep.yaml
  :language: yaml
  :caption: samples/pipeline_template_examples/parameter_sweeps/02_median_filter_kernel_sweep.yaml

What does the output data of a parameter sweep look like?
=========================================================

When a parameter sweep is detected in the process list file, HTTomo will
automatically run that particular method on the "middle slice" of the method's
input data *only*.

.. note:: This means that the parameter sweep will not be performed on the
          *full input data*, but only on a single slice of it.

Meaning, if the method with the parameter sweep takes in projection data, the
method will be run with the different parameter values on the *middle
projection* of the stack. Similarly, for a method that runs on sinogram data,
the method will be run with the different parameter values on the *middle
sinogram* of the stack.

HTTomo produces two outputs for a parameter sweep:

1. An HDF5 file containing multiple datasets

2. A folder of tiff files

Below is some information about how the data is structured in both cases.

Tiff
++++

Output folder
#############

In the output folder of the HTTomo run, let's call it :code:`output_folder/`,
another folder will be created inside. Its name will be
:code:`middle_slices_{DATASET_NAME}/`, where :code:`{DATASET_NAME}` is the name
of the output dataset for the method.

For example, if the method with the parameter sweep was defined to have its
output dataset named :code:`tomo` like so:

.. code-block:: yaml

  - tomopy.misc.corr:
      median_filter:
        data_in: tomo
        data_out: tomo
        size: !Sweep
          - 3
          - 5
        axis: 0

then the folder would be called :code:`middle_slices_tomo/`.

There is then another folder within that, whose name specifies the bit-depth of
the tiff files. Parameter sweeps default to using 8-bit tiffs, so the folder
will be called :code:`images8bit_tif/`.

All in all, the folder containing the tiff files in this case would be called
:code:`output_folder/middle_slices_tomo/images8bit_tif`.

Files
#####

Inside this folder, there would be :code:`n` number of tiff files, where
:code:`n` is the number of values in the parameter sweep. In this case
:code:`n=2`, so there would be two tiff files in the folder
:code:`output_folder/middle_slices_tomo/images8bit_tif`.

The tiff files are named with numbers, such as :code:`00000.tif` and
:code:`00001.tif`. The number in the tiff filename corresponds to the parameter
value position in the sweep. In the above example, the two tiff files would be
called :code:`00000.tif` and :code:`00001.tif`, where

- :code:`00000.tif` contains the result of the method run with the parameter
  value :code:`3`

- :code:`00001.tif` contains the result of the method run with the parameter
  value :code:`5`

HDF5
++++

Suppose the parameter sweep defined is across two values, :code:`3` and
:code:`5`:

.. code-block:: yaml

    size: !Sweep
      - 3
      - 5

Counting from zero, the 0th parameter sweep value would be :code:`3`, and the
1st parameter sweep value would be :code:`5`.

In the HDF5 file output of a parameter sweep, there is one dataset for each
parameter value in the sweep. Each dataset has a path
:code:`/data/param_sweep_{n}` where :code:`{n}` is an integer starting from 0,
representing which value in the parameter sweep it was the result of.

For this particular case, :code:`0 <= {n} <= 1`, and there would be two datasets
in the HDF5 file, :code:`/data/param_sweep_0` and :code:`/data/param_sweep_1`,
where

- :code:`/data/param_sweep_0` contains the result of the method run with the
  parameter value :code:`3`

- :code:`/data/param_sweep_1` contains the result of the method run with the
  parameter value :code:`5`
