.. _create_nxtomo:

Creating an NXtomo file
^^^^^^^^^^^^^^^^^^^^^^^

HTTomo can automatically find tomography data in a `NeXus NXtomo file
<https://manual.nexusformat.org/classes/applications/NXtomo.html>`_. This
tutorial shows how to pack either NumPy arrays or stacks of TIFF images into
that format.

:download:`Download the complete create_nxtomo.py script
<../../scripts/create_nxtomo.py>` before following the examples. The script is
self-contained and may also be imported as a Python module.

Install the packages used by the script in the environment where the packing
will run:

.. code-block:: console

    python -m pip install numpy h5py tifffile

``tifffile`` is only needed for the TIFF example.

Input layout
============

Projection, flat-field, and dark-field data must have the axis order
``(frames, detector_y, detector_x)``. For projections, ``frames`` is the
rotation-angle axis. The angle array must be one-dimensional, measured in
degrees, and contain exactly one value per projection.

A single flat or dark image may be supplied as a two-dimensional array. The
script adds its frame axis automatically. Flats and darks are optional, but
operations that perform flat/dark correction need meaningful calibration
images.

From NumPy arrays
=================

Import :func:`write_nxtomo` from the downloaded script. Arrays can come from
``numpy.load``, another Python library, or calculations performed in the same
program:

.. code-block:: python

    import numpy as np

    from create_nxtomo import write_nxtomo

    projections = np.load("projections.npy")  # (n_angles, detector_y, detector_x)
    angles = np.load("angles.npy")             # (n_angles,), in degrees
    flats = np.load("flats.npy")               # (n_flats, detector_y, detector_x)
    darks = np.load("darks.npy")               # (n_darks, detector_y, detector_x)

    write_nxtomo(
        "scan.nxs",
        projections,
        angles,
        flats=flats,
        darks=darks,
        sample_name="my sample",
        compression="gzip",
    )

Use ``overwrite=True`` only when an existing output file should be replaced.
Set ``compression=None`` for faster writing and a larger file, or
``compression="lzf"`` for lightweight compression. The default is
``compression="gzip"``.

From TIFF stacks
================

Keep each image type in its own directory, for example:

.. code-block:: text

    scan/
    ├── projections/
    │   ├── projection_0000.tif
    │   ├── projection_0001.tif
    │   └── ...
    ├── flats/
    │   ├── flat_0000.tif
    │   └── ...
    ├── darks/
    │   ├── dark_0000.tif
    │   └── ...
    └── angles.npy

Then run the script. Quote each glob so that the script, rather than the
shell, receives and sorts the complete file list:

.. code-block:: console

    python create_nxtomo.py scan.nxs \
        --projections "scan/projections/*.tif" \
        --angles scan/angles.npy \
        --flats "scan/flats/*.tif" \
        --darks "scan/darks/*.tif" \
        --sample-name "my sample"

The angles may instead be stored as a one-column ``.txt`` file or a
comma-separated ``.csv`` file. Omit ``--flats`` or ``--darks`` when that image
type is unavailable. Use ``--overwrite`` to replace an existing output file.

The TIFF loader sorts filenames naturally, so ``projection_2.tif`` precedes
``projection_10.tif``. The filenames still need to encode the acquisition
order correctly. Every matched TIFF must be a single two-dimensional grayscale
image, and all images must have the same height and width.

What the script writes
======================

The output stores darks, then flats, then projections in one three-dimensional
detector dataset. The ``image_key`` gives every frame its NXtomo meaning:

* ``0`` -- projection;
* ``1`` -- flat field; and
* ``2`` -- dark field.

Calibration frames are assigned the first projection angle because NXtomo
requires one rotation angle per frame. HTTomo uses ``image_key`` to select only
the projection frames and their corresponding angles.

The important part of the resulting file is:

.. code-block:: text

    /entry                              NXentry
    ├── definition                     "NXtomo"
    ├── instrument                     NXinstrument
    │   └── detector                   NXdetector
    │       ├── data                   (frames, detector_y, detector_x)
    │       └── image_key              (frames,)
    ├── sample                         NXsample
    │   └── rotation_angle             (frames,), units="deg"
    └── data                           NXdata
        ├── data                       link to detector/data
        ├── image_key                  link to detector/image_key
        └── rotation_angle             link to sample/rotation_angle

The links in ``/entry/data`` do not duplicate the arrays. They expose the
standard NXtomo paths that HTTomo's automatic discovery uses.

Loading the result in HTTomo
============================

Use the standard tomography loader and set all discoverable paths to
``auto``:

.. code-block:: yaml

    - method: standard_tomo
      module_path: httomo.data.hdf.loaders
      parameters:
        data_path: auto
        image_key_path: auto
        rotation_angles: auto

Pass ``scan.nxs`` as the input data file when running HTTomo. If the file has
no flats or darks, HTTomo supplies dummy calibration arrays. Alternatively,
set ``flats: ignore`` or ``darks: ignore`` explicitly when the corresponding
correction should not use stored calibration images; see :ref:`darks_flats`.

Reusable NeXuS writer code
==========================

The core NumPy writer is included below for reference. The downloadable script
also contains TIFF loading, validation, filename sorting, and its command-line
interface.

.. dropdown:: Reusable NeXuS writer code

    .. literalinclude:: ../../scripts/create_nxtomo.py
        :language: python
        :start-after: # [write-nxtomo-start]
        :end-before: # [write-nxtomo-end]
        :linenos:
