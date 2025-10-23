.. _reference_loaders:

HTTomo Loaders
--------------

Available Loaders
=================

HTTomo currently has one loader, :code:`standard_tomo_loader`, which is geared
towards loading "standard" tomography data collected at DLS beamlines.

Basic Usage of the Standard Loader
==================================

This loader has several parameters which are fairly self-explanatory. Other
parameters either require some more detail to use, or have extra capabilities which
are not obvious if one has seen only one or two simple loader configurations.

The following YAML configuration of the loader shows all the parameters needed, and
is the standard case (ie, there's no configuration in it for special cases), so it
serves as a good starting point:

.. code-block:: yaml

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        data_path: /entry1/tomo_entry/data/data
        image_key_path: /entry1/tomo_entry/instrument/detector/image_key
        rotation_angles:
          data_path: /entry1/tomo_entry/data/rotation_angle

.. note:: The input data being loaded is assumed to be in hdf5/NeXuS file format,
   in accordance with the data typically collected at a DLS beamline.

.. _nxtomo_discovery:

Automatic `NXtomo` Discovery
++++++++++++++++++++++++++++

If the input file has a valid `NXtomo` entry (see the `NXtomo application
definition <https://manual.nexusformat.org/classes/applications/NXtomo.html>`_
for more details) then the loader can be configured to automatically discover
it, without needing to explicitly specify values like the dataset path.

This configuration is done by providing the :code:`auto` value to the following
parameters:

- :code:`data_path`
- :code:`image_key_path`
- :code:`rotation_angles`

For example:

.. code-block:: yaml

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        data_path: auto
        image_key_path: auto
        rotation_angles: auto

.. note:: Automatic :code:`NXtomo` discovery (and therefore the :code:`auto` value) is not
   supported when the darks/flats are separate from the projection data.

Manually Providing Dataset Paths
++++++++++++++++++++++++++++++++

:code:`data_path`
~~~~~~~~~~~~~~~~~

The :code:`data_path` parameter is the path to the dataset in the input hdf5/NeXuS
file containing the image data (usually, projections + darks + flats are in the
same dataset).

:code:`image_key_path`
~~~~~~~~~~~~~~~~~~~~~~

The :code:`image_key_path` parameter is the path to the dataset in the input
hdf5/NeXuS file containing the so called "image key".

.. note:: The "image key" is an array whose length is the same as the number of
   images in the collected data. Each element has a value of 0, 1, or 2, to
   indicate a projection (0), flat field image (1), or dark-field image (2).

:code:`rotation_angles`
~~~~~~~~~~~~~~~~~~~~~~~

Typically, the rotation angle values are stored in a dataset within the input
hdf5/NeXuS file. This dataset is usually what is provided to serve as the
rotation angle values during processing.

In such cases, the :code:`rotation_angles` parameter has two lines of
configuration. Specifying :code:`data_path` is meaning that the rotation angles are
indeed stored in a dataset within the input hdf5/NeXuS file, and the given path is
the path to that dataset within the input hdf5/NeXuS file.

The reason the :code:`rotation_angles` parameter isn't simply one value (a path)
like the previously mentioned parameters is because there are situations when the
angles dataset in the input hdf5/NeXus file doesn't exist, or cannot be used.

To configure the loader to handle such cases, please refer to
:ref:`user_defined_angles`.


Dealing with darks and flats
============================

HTTomo currently supports several options to deal with the flats and darks images.

Files that do not contain image keys
++++++++++++++++++++++++++++++++++++

These are the files without the image keys that contain only flats or darks in two separate files.
Here one needs to add :code:`darks` and :code:`flats` parameters to the loader parameters with the following fields (see the example bellow): 

- :code:`file`, the path to the hdf5/NeXus file containing the darks/flats
- :code:`data_path`, the dataset within the hdf5/NeXus file that contains the
  darks/flats

.. code-block:: yaml
   :emphasize-lines: 5,6,8,9

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        darks:
          file: path/to/new/file.nxs
          data_path: /entry1/tomo_entry/data/data
        flats:
          file: path/to/new/file.nxs
          data_path: /entry1/tomo_entry/data/data

Files with image keys
+++++++++++++++++++++

This can be the case when a new scan is performed, which contains the required image keys. Therefore the keys
in the older scan should be ignored. In this instance, we need to provide a parameter :code:`image_key_path` in addition to 
:code:`file` and :code:`data_path` fields.

.. code-block:: yaml
   :emphasize-lines: 7,11


    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        darks:
          file: path/to/new/file.nxs
          data_path: /entry1/tomo_entry/data/data
          image_key_path: /entry1/tomo_entry/instrument/detector/image_key
        flats:
          file: path/to/new/file.nxs
          data_path: /entry1/tomo_entry/data/data
          image_key_path: /entry1/tomo_entry/instrument/detector/image_key 
        
.. _user_defined_angles:

Data without darks/flats
++++++++++++++++++++++++

It is also possible to process the data that does not contain darks or flats, i.e., the pipeline runs without given darks or flats.
Nothing specific should be done about it in the loader, it will be handled automatically without any extra configuration needed.   

Ignore darks/flats
++++++++++++++++++

This is the case when darks or flats still present in the dataset, but one needs to ignore either of them or both of them. This can be done by providing  
the keyword :code:`ignore` into the loader, like in the example below where both flats and darks are ignored:

.. code-block:: yaml
   :emphasize-lines: 4-5


    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        darks: ignore
        flats: ignore


Providing/Overriding Angles Data
================================

There are several situations in which overriding the angles dataset in the input
hdf5/NeXuS file, or generating an array due to the absence of an angles dataset, is
necessary. The loader offers the ability to specify an angles array via:

- start angle
- stop angle
- total number of angles

values by configuring the :code:`rotation_angles` parameter slightly differently
than shown earlier.

The following is reusing the same example from the separate darks/flats example,
but is now drawing attention to the :code:`rotation_angles` parameter:


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

It can be seen that :code:`user_defined` has been specified instead of
:code:`data_path`. Furthermore, there are then three fields provided:

- :code:`start_angle`, which is the first angle (in degrees)
- :code:`stop_angle`, which is the last angle (in degrees)
- :code:`angles_total`, which is the number of angles on total to have in that
  range, equally spaced

to generate the desired angles array that HTTomo will use during pipeline
execution.

Previewing
==========

The data being loaded with the loader can be cropped/previewed prior to being
passed along to the first method. The loader has the :code:`preview` parameter
for configuring the cropping/previewing. Please see :ref:`previewing` for more
details on previewing.
