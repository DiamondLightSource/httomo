.. _user_defined_angles:

Rotation angles
^^^^^^^^^^^^^^^

The standard loader normally reads rotation angles from the input file:

.. code-block:: yaml

    rotation_angles:
      data_path: /entry1/tomo_entry/data/rotation_angle

If this dataset is absent or unsuitable, define an evenly spaced angle array
using its start, stop, and total number of angles:

.. code-block:: yaml
   :emphasize-lines: 8-10

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        data_path: /1-TempPlugin-tomo/data
        image_key_path: /entry1/tomo_entry/instrument/detector/image_key
        rotation_angles:
          user_defined:
            start_angle: 0
            stop_angle: 180
            angles_total: 724

``start_angle`` and ``stop_angle`` are measured in degrees.
``angles_total`` specifies how many equally spaced angles HTTomo generates.
