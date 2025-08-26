.. _save-result-examples:

Saving intermediate files
+++++++++++++++++++++++++

HTTomo, by default, will *not* write the output of a method to a file unless under certain conditions (please see more in :ref:`httomo-saving`).

HTTomo can be informed to write or not write the output of a method to a file
with the :code:`save_result` parameter. Its value is a boolean, so either
:code:`true` or :code:`false` are valid values for it.


Example 1: save output of a specific method
###########################################

Suppose we wanted to save the output of the normalisation function :code:`normalize`. Then we
should add :code:`save_result: true` to the list of the function parameters, but NOT the method's parameters:

.. code-block:: yaml
  :emphasize-lines: 8

  - method: normalize
    module_path: httomolibgpu.prep.normalize
    parameters:
      cutoff: 10.0
      minus_log: true
      nonnegativity: false
      remove_nans: false
    save_result: true

Example 2: using :code:`--save_all` and :code:`save_result` together
####################################################################

When the :code:`--save_all` option/flag is provided, the :code:`save_result`
parameter can be used to override individual method's to *not* save their
output.

In contrast to the previous example, suppose we had a process list where we
would like to save the output of all methods using :code:`--save_all`, *apart* from the
:code:`normalize` method.

.. code-block:: yaml
  :emphasize-lines: 8

  - method: normalize
    module_path: httomolibgpu.prep.normalize
    parameters:
      cutoff: 10.0
      minus_log: true
      nonnegativity: false
      remove_nans: false
    save_result: false