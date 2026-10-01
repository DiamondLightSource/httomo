Standard tomography loader
^^^^^^^^^^^^^^^^^^^^^^^^^^

HTTomo provides the :code:`standard_tomo` loader for parallel-beam tomography
data stored in HDF5/NeXus files. A basic configuration specifies the projection
data, image keys, and rotation angles:

.. code-block:: yaml

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        data_path: /entry1/tomo_entry/data/data
        image_key_path: /entry1/tomo_entry/instrument/detector/image_key
        rotation_angles:
          data_path: /entry1/tomo_entry/data/rotation_angle

``data_path``
   The dataset containing the projection data. It commonly also contains the
   dark and flat images.

``image_key_path``
   The dataset identifying each image as a projection (0), flat (1), or dark
   (2).

``rotation_angles``
   The location or definition of the rotation angles. See
   :ref:`user_defined_angles` when the input does not contain a usable angle
   dataset.

See :ref:`darks_flats` for other dark and flat layouts, and :ref:`previewing`
for loading only part of the dataset.

.. _nxtomo_discovery:

Automatic NXtomo discovery
==========================

If the input contains a valid `NXtomo entry
<https://manual.nexusformat.org/classes/applications/NXtomo.html>`_, set the
following parameters to :code:`auto`:

.. code-block:: yaml

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        data_path: auto
        image_key_path: auto
        rotation_angles: auto

HTTomo then discovers the projection data, image keys, and rotation angles. The NXtomo compatible data can be created using the script from :ref:`create_nxtomo`.

.. note:: Automatic NXtomo discovery is unavailable when darks or flats are
   loaded from separate files.
