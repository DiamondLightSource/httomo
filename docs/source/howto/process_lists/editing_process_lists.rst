Editing process lists
---------------------

The concepts in the previous page will have hopefully helped you to understand
how to define your desired pipeline with HTTomo in the most computationally
efficient manner. This page will provide guidance on editing a process list
file.

Workflow for Writing Process Lists
==================================

Given time working with HTTomo, a user will likely settle on a workflow for
defining process list YAML files that suits their individual needs.

As a starting point, something like the following is suggested:

- copy+paste templates for the desired methods from the
  :ref:`reference_templates` section
- manually edit the parameter values within the copied template as needed,
  checking the documentation for the relevant backend if/when necessary for
  further guidance on the method's parameters
- intermittently run the :ref:`YAML checker <utilities_yamlchecker>` during
  editing of the YAML file to detect any errors early on

Method Parameters vs. HTTomo Method Parameters
==============================================

In addition to parameters for a method that influence the processing that the
method performs, there are some parameters that can be used in a method's YAML
configuration that modify in some way how HTTomo handles the method (for
example, whether or not to write the output of the method to a file).

Since these parameters are *not* parameters for the method itself and are
instead specific to HTTomo, the next section will go through these extra
parameters, explaining what they are for and giving examples of how they can be
used.

HTTomo Method Parameter Guide
=============================

.. _save-result-examples:

Saving intermediate files with :code:`save_result`
++++++++++++++++++++++++++++++++++++++++++++++++++

As explained in :ref:`httomo-saving`, by default, HTTomo will *not* write the
output of a method to a file unless under certain conditions (please see the
link for a description of these file-saving conditions).

HTTomo can be informed to write or not write the output of a method to a file
with the :code:`save_result` parameter. Its value is a boolean, so either
:code:`True` or :code:`False` are valid values for it.


Example: save output of a specific method
#########################################

Suppose we wanted to save the output of the TomoPy :code:`median_filter`, we
could add :code:`save_result: True` to its parameters:

.. code-block:: yaml
  :emphasize-lines: 7

  - tomopy.misc.corr:
      median_filter:
        data_in: tomo
        data_out: tomo
        size: 3
        axis: 0
        save_result: True

Example: using :code:`--save_all` and :code:`save_result` together
##################################################################

When the :code:`--save_all` option/flag is provided, the :code:`save_result`
parameter can be used to override individual method's to *not* save their
output.

In contrast to the previous example, suppose we had a process list where we
would like to save the output of all methods *apart* from the
:code:`median_filter`.  This could be achieved by using :code:`--save_all` when
running HTTomo, along with providing :code:`save_result: False` for
:code:`median_filter` in the YAML:

.. code-block:: yaml
  :emphasize-lines: 7

  - tomopy.misc.corr:
      median_filter:
        data_in: tomo
        data_out: tomo
        size: 3
        axis: 0
        save_result: False

Omitting :code:`data_out`
+++++++++++++++++++++++++

It could be the case that one would like to have the same name for the output
dataset as the input dataset (keeping in mind that this will *overwrite* the
input of a method with the output of the method once the method has finished its
processing).

In such cases, it can be tedious or an eyesore in the YAML to always be
specifying both :code:`data_in` and :code:`data_out` to have the same value in
each method's configuration.

To help with this, the :code:`data_out` parameter can be omitted and HTTomo will
assume that the output dataset name is the same as the input dataset name.

.. warning::
  As previously stated, please be aware that using this will be *overwriting*
  the input dataset of a method with the output of the method.

  In particular, please be careful and check if any methods further down the
  pipeline would need to process the original dataset, as this will influence if
  using this functionality will unintentionally interfere with the desired
  pipeline.

Processing multiple inputs with :code:`data_in_multi` and :code:`data_out_multi`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Sometimes it can be useful to process multiple datasets with the same method +
parameters. In principle, this could be done by duplicating the method
configuration for each dataset needing to be processed, and then changing the
:code:`data_in` parameter for each duplicated block.

However, HTTomo provides a more concise way of achieving the same result with
only *one* copy of the method configuration. This is done by using the
:code:`data_in_multi` and :code:`data_out_multi` parameters instead of
:code:`data_in` and :code:`data_out` respectively.

Instead of providing a single input/output dataset name, a *list* of dataset
names can be given to :code:`data_in_multi` and :code:`data_out_multi`.

The list of dataset names given to :code:`data_in_multi` should specify the
different datasets that you would like to be processed by the method. Similarly,
the list of dataset names given to :code:`data_out_multi` should specify what
you would like the corresponding ouptut datat names to be.

Example: processing three datasets with the same method
#######################################################

A snippet take from an example pipeline in the HTTomo repo shows
:code:`median_filter` from TomoPy being applied to three datasets: projections,
darks, and flats:

.. literalinclude:: ../../../../samples/pipeline_template_examples/multi_inputs/01_multi_inputs.yaml
  :language: yaml
  :lines: 6-11
  :emphasize-lines: 3-4
  :caption: samples/pipeline_template_examples/multi_inputs/01_multi_inputs.yaml

As a variation on this same example: in a similar fashion to how
:code:`data_out` can be omitted and the output dataset is assumed to be the same
as :code:`data_in`, it is the case that :code:`data_out_multi` can be omitted
and the output datasets are assumed to be the same as :code:`data_in_multi`. So,
the below achieves the same as above:

.. code-block:: yaml
  :emphasize-lines: 3

  - tomopy.misc.corr:
      median_filter:
        data_in_multi: [tomo, flats, darks]
        size: 3
        axis: 0

Example: using different names for the output datasets
######################################################

In principle, the output dataset names need not be the same as the input dataset
names. Different dataset names are valid, so the following is fine:

.. code-block:: yaml
  :emphasize-lines: 3-4

  - tomopy.misc.corr:
      median_filter:
        data_in_multi: [tomo, flats, darks]
        data_out_multi: [tomo_diff, flats_diff, darks_diff]
        size: 3
        axis: 0

Using a dataset as a parameter value
++++++++++++++++++++++++++++++++++++

There are cases where the output dataset of a method is needed as the value of
a parameter for a method further down the pipeline. For example, the output of a
method that calculates the center of rotation needing would be used as the value
of a CoR parameter for a reconstruction method.

HTTomo supports providing a dataset name as the value of a parameter to handle
situations like this.

Example
#######

The following example comes from taking snippets from an example in the HTTomo
repo, where the :code:`find_center_vo` method is generating an output dataset
called :code:`cor`, which is then being used as the value of the :code:`center`
parameter for the :code:`recon` method:

.. literalinclude:: ../../../../samples/pipeline_template_examples/02_basic_cpu_pipeline_tomo_standard.yaml
  :language: yaml
  :lines: 31-50
  :emphasize-lines: 4,16
  :caption: samples/pipeline_template_examples/02_basic_cpu_pipeline_tomo_standard.yaml
