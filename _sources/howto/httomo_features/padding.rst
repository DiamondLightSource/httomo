.. default-role:: math
.. _padding:

Padding
^^^^^^^

HTTomo processes data as :ref:`chunks_data` and :ref:`blocks_data`. Methods
that operate on independent 2D frames, such as 2D denoising filters, do not
need :term:`padding`. Methods that operate on 3D volumes need padded blocks to preserve
boundary conditions and prevent artefacts.

How this can be useful?
=======================

Padding enables fully 3D methods, which can provide consistent resolution in
all dimensions, improve contrast, and remove artefacts more effectively.

.. list-table::

    * - .. figure:: ../../_static/padding/denoising2d.jpg
           :scale: 20 %

           2D denoising produces inconsistent vertical resolution.

      - .. figure:: ../../_static/padding/denoising3d_pad5.jpg
           :scale: 20 %

           3D denoising produces consistent resolution in all dimensions.

.. note:: Padding supports advanced 3D filters and iterative reconstruction
   methods.

How to use
==========

There is no need to do anything specific to enable this feature as it will
be switched on automatically when the method that requires it added to the
pipeline.
