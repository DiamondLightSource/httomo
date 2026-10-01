.. _save-result-examples:

Save intermediate datasets
++++++++++++++++++++++++++

Use the top-level ``save_result`` field to control whether a method's processed
dataset is written to an intermediate HDF5 file. Its value is a YAML Boolean:
``true`` or ``false`` without quotation marks.

.. warning::

  Place ``save_result`` beside ``method``, ``module_path`` and ``parameters``,
  not inside the ``parameters`` mapping.

For example, save the result of ``dark_flat_field_correction`` as follows:

.. code-block:: yaml
   :emphasize-lines: 6

    - method: dark_flat_field_correction
      module_path: httomolibgpu.prep.normalize
      parameters:
        flats_multiplier: 1.0
        darks_multiplier: 1.0
      save_result: true

If ``save_result`` is omitted, the method's default setting is used. To save
intermediate datasets for every applicable task, run HTTomo with
``--save-all``. When that option is enabled, ``save_result: false`` *does not
disable* saving for an individual method.

.. note::

  Saving intermediate datasets can substantially increase disk use and execution time, so enable it only for results that need to be inspected or reused. The result of any reconstruction method is saved by default into an intermediate file.

For other output choices, use a ``save_to_images`` method to write images or
``--save-snapshots`` to capture lightweight diagnostic snapshots. See
:ref:`reference_templates` and :ref:`run-httomo-indepth` respectively.
