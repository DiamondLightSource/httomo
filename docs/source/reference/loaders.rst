.. _reference_loaders:

HTTomo Loaders
--------------

Ignoring darks, flats, or projections
=====================================

There are situations where certain dark field, flat field, or projection images
that have been collected in the data need to be excluded from processing. The
standard tomography loader in HTTomo offers the functionality to exclude dark
field, flat field, and projection images based on their index in the dataset
they are contained in. The images can be excluded by specifying either:

- Individual indices
- A range of indices with start and stop values

This functionality is used in a process list via the :code:`ignore_darks`,
:code:`ignore_flats`, and :code:`ignore_projections` parameters.

Ignore modes
============

To specify individual indices to exclude, use the :code:`individual` subfield,
and to specify a range of indices to exclude, use the :code:`batch` subfield.
These subfields can both be used at the same time to select batches of
darks/flats/projections as well as individual darks/flats/projections to ignore.

.. note:: The range provided to the :code:`batch` option via :code:`start` and
          :code:`stop` is *inclusive*; ie, the :code:`stop` value is *included*
          in the index values to ignore.

It is also possible to ignore *all* darks and/or flats by setting the value of
either :code:`ignore_darks` or :code:`ignore_flats` to :code:`true`. However,
ignoring all *projections* is not yet supported.

The following is a summary of the supported modes of ignoring images for the
cases of darks, flats, and projections:

============ ========= ========= ===============
Ignore mode  Darks     Flats     Projections
============ ========= ========= ===============
Individual   Supported Supported Supported
Batch        Supported Supported Supported
All          Supported Supported *Not supported*
============ ========= ========= ===============

.. note:: Ignoring projections is only supported when reading the data in as
          projections; ie, when :code:`--dimension 1` is used when running
          HTTomo (which is the default value when the flag is not explicitly
          passed)

Examples
========

Below are some example uses of the :code:`ignore_darks` parameter, and the
:code:`ignore_flats` parameter can be used in a similar manner. The ignoring of
individual and batch projections is the same as well.

Ignore individual darks only
++++++++++++++++++++++++++++

.. code-block:: yaml
  :emphasize-lines: 12-13

    - httomo.data.hdf.loaders:
        standard_tomo:
        name: tomo
        data_path: entry1/tomo_entry/data/data
        image_key_path: entry1/tomo_entry/instrument/detector/image_key
        dimension: 1
        preview:
            -
            - start: 30
              stop: 60
            -
        ignore_darks:
            individual: [0, 5]

Ignore all darks
++++++++++++++++

.. code-block:: yaml
  :emphasize-lines: 12

    - httomo.data.hdf.loaders:
        standard_tomo:
        name: tomo
        data_path: entry1/tomo_entry/data/data
        image_key_path: entry1/tomo_entry/instrument/detector/image_key
        dimension: 1
        preview:
            -
            - start: 30
              stop: 60
            -
        ignore_darks: true

Ignore individual darks and a group of darks
++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: yaml
  :emphasize-lines: 12-16

    - httomo.data.hdf.loaders:
        standard_tomo:
        name: tomo
        data_path: entry1/tomo_entry/data/data
        image_key_path: entry1/tomo_entry/instrument/detector/image_key
        dimension: 1
        preview:
            -
            - start: 30
              stop: 60
            -
        ignore_darks:
          individual: [0, 3]
          batch:
            - start: 5
              stop: 10


Ignore two groups of darks
++++++++++++++++++++++++++

.. code-block:: yaml
  :emphasize-lines: 12-17

    - httomo.data.hdf.loaders:
        standard_tomo:
        name: tomo
        data_path: entry1/tomo_entry/data/data
        image_key_path: entry1/tomo_entry/instrument/detector/image_key
        dimension: 1
        preview:
            -
            - start: 30
              stop: 60
            -
        ignore_darks:
          batch:
            - start: 5
              stop: 10
            - start: 20
              stop: 30
