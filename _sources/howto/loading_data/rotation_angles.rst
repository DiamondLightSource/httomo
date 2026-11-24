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

.. note::

   Both ``start_angle`` and ``stop_angle`` are included in the generated array.
   For more than one angle, the spacing in degrees is
   ``(stop_angle - start_angle) / (angles_total - 1)``.

   For example, a limited-angle scan from -49 to 59 degrees inclusive, with a
   step of 0.05 degrees, contains 2161 projections:

   .. code-block:: yaml

       rotation_angles:
         user_defined:
           start_angle: -49
           stop_angle: 59
           angles_total: 2161

   Set ``angles_total`` to the number of projection images, excluding flat and
   dark frames.
