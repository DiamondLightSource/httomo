.. _reference_loaders:
.. _loading_data:

Loading data
************

HTTomo's standard tomography loader reads projection data, darks, flats, and
rotation angles from HDF5/NeXus files. It can discover NXtomo datasets
automatically, combine data from separate files, crop the input, and select an
individual scan from a continuous acquisition.

.. toctree::
   :maxdepth: 2

   loading_data/standard_loader
   loading_data/darks_flats
   loading_data/rotation_angles
   loading_data/previewing
   loading_data/continuous_scan_subset
   loading_data/create_nxtomo
