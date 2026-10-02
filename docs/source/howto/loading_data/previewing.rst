.. default-role:: math
.. _previewing:

Previewing
^^^^^^^^^^

Previewing crops, or slices, the input data. It can remove unused regions and
reduce processing time, particularly while :ref:`sweeping parameters
<parameter_sweeping>`. See :ref:`previewing_enable` to start configuring it.

Previewing in the loader
========================

The :doc:`standard_loader` provides a :code:`preview` parameter for selecting
part of the input data.

.. note:: HTTomo assumes a three-dimensional array whose axes are the *angular*
   dimension, the vertical detector (`Y`) and the horizontal detector (`X`),
   in that order (see :numref:`fig_dimsdata`).

.. _fig_dimsdata:
.. figure::  ../../_static/preview/dims_prev.svg
    :scale: 55 %
    :alt: 3D data

    3D projection data and their axes


The :code:`preview` parameter
=============================

The :term:`preview` parameter has one field per axis. Each field accepts
:code:`start` and :code:`stop` values:

.. code-block:: yaml

   preview:
    angles:
      start:
      stop:
    detector_y:
      start:
      stop:
    detector_x:
      start:
      stop:

The stop value is excluded, as in a Python slice. For example, the following
selection loads projections 20 through 99:

.. code-block:: yaml

   preview:
     angles:
       start: 20
       stop: 100

.. note::

   ``continuous_scan_subset`` also selects the angular range. When it is set,
   it replaces ``preview.angles``. The command-line
   ``--continuous-scan-subset`` option takes precedence over both values. See
   :ref:`continuous_scan_subset_selection`.

Using the full dataset
======================

Omitting :code:`preview` selects the full dataset without cropping.

.. _previewing_enable:

Enabling data preview
=====================

Crop either or both detector dimensions to reduce the data size and accelerate
processing.

.. note:: Removing blank detector regions reduces the reconstructed volume and
   can also accelerate post-processing.

The following projections show vertical and horizontal cropping.

Before cropping |pic1| and after |pic2|

.. |pic1| image:: ../../_static/preview/uncropped.gif
   :width: 44%

.. |pic2| image:: ../../_static/preview/cropped.gif
   :width: 27%


1. Crop blank regions from the top and bottom of the vertical detector (`Y`),
   as shown in :numref:`fig_dimsdataY`. Inspect the raw projections to identify
   regions that remain blank throughout the scan.

   .. code-block:: yaml

       preview:
         detector_y:
           start: 200
           stop: 1800

   This selects slices 200 to 1799, producing a vertical dimension of 1600
   pixels. The equivalent Python slice is :code:`[:, 200:1800, :]`.

.. _fig_dimsdataY:
.. figure::  ../../_static/preview/dims_prevY.svg
    :scale: 55 %
    :alt: 3D data, Y slicing

    Cropping detector- `Y` dimension of 3D projection data

2. Crop blank regions from the left and right of the horizontal detector (`X`),
   as shown in :numref:`fig_dimsdataX`.

   .. warning::
      Horizontal cropping can disrupt automatic centering and introduce
      reconstruction artefacts, particularly with iterative methods. Crop the
      `X` dimension conservatively.

   .. code-block:: yaml

       preview:
         detector_x:
           start: 100
           stop: 2000

   The equivalent Python slice is :code:`[:, :, 100:2000]`.

.. _fig_dimsdataX:
.. figure::  ../../_static/preview/dims_prevX.svg
    :scale: 55 %
    :alt: 3D data, X slicing

    Cropping detector- `X` dimension of 3D projection data

Combine both operations as follows:

.. code-block:: yaml

    preview:
      detector_y:
        start: 200
        stop: 1800
      detector_x:
        start: 100
        stop: 2000

Using :code:`begin`, :code:`mid` and :code:`end` with offsets
================================================================

Use :code:`begin`, :code:`mid` and :code:`end` instead of absolute indices
when the input dimensions are unknown. They may be used in the angular range
as well as the detector ranges. Adjust them with :code:`start_offset` and
:code:`stop_offset`:

.. code-block:: yaml

    preview:
      detector_x:
        start: begin
        start_offset: 100
        stop: end
        stop_offset: -100
      detector_y:
        start: mid
        start_offset: -50
        stop: mid
        stop_offset: 50

This removes 100 pixels from each end of :code:`detector_x`, equivalent to
:code:`[100:-100]`, and selects 100 pixels centred on :code:`detector_y`.

.. note:: :code:`begin`, :code:`mid` and :code:`end` identify the first,
   middle and last indices of a dimension, respectively.


Using :code:`mid` by itself
===========================

The :code:`detector_y` and :code:`detector_x` fields also accept :code:`mid`
without :code:`start` or :code:`stop`:

.. code-block:: yaml

    preview:
      detector_y:
        mid

This selects the middle three slices of the specified dimension.

.. warning:: The :code:`angles` field does not support :code:`mid`.

Omitting :code:`preview` fields
===============================

You may omit unused dimension fields and :code:`start` or :code:`stop` values.

Omitting one or more dimension fields
-------------------------------------

An omitted or empty dimension field selects that entire dimension. The following
configuration therefore selects the full dataset:

.. code-block:: yaml

    preview:
      angles:
      detector_y:
      detector_x:

Omitting the :code:`start` or :code:`stop` fields
-------------------------------------------------

For each dimension:

- Omitting :code:`start` begins at index 0.
- Omitting :code:`stop` continues to the end of the dimension.
