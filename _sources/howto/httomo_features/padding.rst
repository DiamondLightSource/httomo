.. default-role:: math
.. _padding:

Padding
^^^^^^^

Padding is an important feature of HTTomo when performing computations on :ref:`chunks_data` and :ref:`blocks_data`.
If the method is a 2D method (work with 2D frames), e.g., a denoising filter, then the data does not require any padding as 
the boundary conditions between blocks will not be violated. However, when the method is a 3D method and it works 
with 3D volumes, then in order to satisfy boundary conditions and avoid artefacts, one needs to pad blocks. 

How this can be useful?
=======================

It is useful because when padding feature exists one can use true fully 3D methods which provide a consistent resolution in
all three dimensions, see the image below. Because of the access to 3D data, one can perform better in removing artefacts,
improving contrast, etc. 

.. list-table:: 

    * - .. figure:: ../../_static/padding/denoising2d.jpg
           :scale: 20 %

           2D denoising applied to 3D data, note the resolution inconsistency in the vertical direction.

      - .. figure:: ../../_static/padding/denoising3d_pad5.jpg 
           :scale: 20 %

           3D denoising applied to 3D data. The resolution is consistent in all three dimensions.

.. note:: With padding enabled, HTTomo can perform more state-of-the-art filtering techniques as well as advanced iterative reconstruction in 3D.