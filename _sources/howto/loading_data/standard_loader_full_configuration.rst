
.. _full_tomo_loader_configuration:

Full loader configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The example below combines the loader options described in the preceding
sections. It assumes that:

* The input file contains a dataset of projections only.
* Image keys and rotation angles are not available in the input file.
* Dark and flat images are stored in separate HDF5 files.

Replace the file paths, dataset paths and rotation angle values with those
appropriate for your data. The example angle values describe 2161 projections
acquired from -49 to 59 degrees in steps of 0.05 degrees.

.. code-block:: yaml

   - method: standard_tomo
     module_path: httomo.data.hdf.loaders
     parameters:
       data_path: /path/to/data
       image_key_path: null # no keys in the file
       rotation_angles: # in degrees
         user_defined:
           start_angle: -49
           stop_angle: 59
           angles_total: 2161
       preview:
         detector_x:
           start: null
           stop: null
         detector_y:
           start: null
           stop: null
       darks:
         file: path/to/file_darks.h5
         image_key_path: null
         data_path: /path/to/darks
       flats:
         file: path/to/file_flats.h5
         image_key_path: null
         data_path: /path/to/flats
       continuous_scan_subset: null

.. note::

   User-defined rotation angles are equally spaced and expressed in degrees.
   Both ``start_angle`` and ``stop_angle`` are included. ``angles_total``
   must match the number of projections, excluding dark and flat images.

   With ``image_key_path: null``, every image in the projection dataset is
   treated as a projection. The dataset must therefore not contain interleaved
   dark or flat images.

.. tip::

   Use ``file: input_data`` if the dark or flat dataset is stored in the main
   input file. Dataset paths such as ``/path/to/flats`` identify datasets
   inside an HDF5 file; they are not filesystem paths.

   Null preview (``null``) bounds select the full detector extent. This example does
   not specify ``preview.angles`` and therefore retains all projections.
   ``continuous_scan_subset: null`` disables configuration-based selection
   of an individual scan from a continuous acquisition.