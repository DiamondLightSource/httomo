.. _darks_flats:

Darks and flats
^^^^^^^^^^^^^^^

The standard loader supports dark and flat images that are:

* stored with the projections;
* stored in separate files or datasets;
* absent; or
* present but intentionally ignored.

Stored with the projections
===========================

When projections, darks, and flats share one dataset, use
:code:`image_key_path` to identify their image types. This is the standard
configuration shown in :doc:`standard_loader`.

Separate datasets without image keys
=====================================

If a separate dataset contains only darks or only flats, specify its
:code:`file` and :code:`data_path`:

.. code-block:: yaml
   :emphasize-lines: 5,6,8,9

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        darks:
          file: path/to/darks.nxs
          data_path: /entry1/tomo_entry/data/data
        flats:
          file: path/to/flats.nxs
          data_path: /entry1/tomo_entry/data/data

Use :code:`input_data` when the separate datasets are in the main input file:

.. code-block:: yaml
   :emphasize-lines: 5,8

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        darks:
          file: input_data
          data_path: /exchange/darks
        flats:
          file: input_data
          data_path: /exchange/flats

Separate data with image keys
=============================

If a specified dataset contains several image types, also provide its
:code:`image_key_path`:

.. code-block:: yaml
   :emphasize-lines: 7,11

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        darks:
          file: path/to/darks.nxs
          data_path: /entry1/tomo_entry/data/data
          image_key_path: /entry1/tomo_entry/instrument/detector/image_key
        flats:
          file: path/to/flats.nxs
          data_path: /entry1/tomo_entry/data/data
          image_key_path: /entry1/tomo_entry/instrument/detector/image_key

Missing darks or flats
======================

No additional configuration is required when the data does not contain darks
or flats. HTTomo handles their absence automatically.

Ignoring darks or flats
=======================

Set either parameter to :code:`ignore` to exclude images that are present in
the dataset:

.. code-block:: yaml
   :emphasize-lines: 4,5

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        darks: ignore
        flats: ignore
