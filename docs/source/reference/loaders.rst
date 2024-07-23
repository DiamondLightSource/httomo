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

:code:`data_path`
+++++++++++++++++

The :code:`data_path` parameter is the path to the dataset in the input hdf5/NeXuS
file containing the image data (usually, projections + darks + flats are in the
same dataset).

:code:`image_key_path`
++++++++++++++++++++++

The :code:`image_key_path` parameter is the path to the dataset in the input
hdf5/NeXuS file containing the so called "image key".

.. note:: The "image key" is an array whose length is the same as the number of
   images in the collected data. Each element has a value of 0, 1, or 2, to
   indicate a projection (0), flat field image (1), or dark-field image (2).

:code:`rotation_angles`
+++++++++++++++++++++++

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

Data with Separate Darks and/or Flats
=====================================

It can sometimes be the case that darks and flats are written to separate
hdf5/NeXuS files, rather than written to the same hdf5/NeXus file as the
projections.

Omitting the image key
++++++++++++++++++++++

In such cases, there is no image key dataset in the hdf5/NeXus file containing the
projections (because there are only projections in the dataset, rather than
projections + darks + flats, so there's no need to have an image key). Due to this,
one difference to the previously shown configuration to handle this case is that
the :code:`image_key_path` parameter is omitted.

Loading the separate darks and flats
++++++++++++++++++++++++++++++++++++

Additionally, there is a need to specify:

- the path to the hdf5/NeXuS file containing the darks/flats
- the dataset within the given hdf5/NeXus file that contains the darks/flats data

In order to specify this information for both darks and flats, there is the
:code:`darks` and :code:`flats` parameters, see the following as an example:

.. literalinclude:: ../../../tests/samples/pipeline_template_examples/DLS/03_i12_separate_darks_flats.yaml
   :language: yaml
   :emphasize-lines: 14-19

Both parameters have two fields that needs to be specified:

- :code:`file`, the path to the hdf5/NeXus file containing the darks/flats
- :code:`data_path`, the dataset within the hdf5/NeXus file that contains the
  darks/flats

.. _user_defined_angles:

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


.. literalinclude:: ../../../tests/samples/pipeline_template_examples/DLS/03_i12_separate_darks_flats.yaml
   :language: yaml
   :emphasize-lines: 7-11

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
