.. _info_reslice:

Re-slicing
==========

A *re-slice* changes how tomographic data is divided and accessed. It is required
when consecutive :ref:`sections <info_sections>` use different data patterns,
typically when switching between projections and sinograms.

.. _fig_reslice:

.. figure:: ../../_static/reslice_gather/httomo_reslicing_dark.png
   :alt: Projection data re-sliced and redistributed into sinogram-oriented chunks
   :align: center
   :width: 100%

   Projection-oriented chunks are re-sliced and redistributed into
   sinogram-oriented chunks.

How re-slicing works
~~~~~~~~~~~~~~~~~~~~

Each sinogram is formed by selecting the same detector row from every projection
angle. HTTomo redistributes these rows among the MPI processes to create
sinogram-oriented chunks.

Re-slicing is performed in CPU memory, so data held on GPUs must first be
transferred back to the host. If sufficient CPU memory is unavailable, HTTomo uses
temporary disk storage instead.

Performance considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Re-slicing can be expensive for large datasets, particularly when temporary disk
storage is required. Pipelines should therefore group methods by data pattern to
minimise pattern changes. A typical preprocessing and reconstruction pipeline
requires only one transition from projections to sinograms.