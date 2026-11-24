.. _real-data-distortion:

Distortion-correction data
==========================

This example reconstructs a 180-degree scan from `Tomographic data for
demonstrating distortion correction methods <distortion-dataset_>`_,
published on Zenodo by Nghia T. Vo, Robert C. Atwood and Michael Drakopoulos.

Detector optics can displace image features and introduce shape errors and
streak artefacts in a reconstruction. Dark/flat-field correction adjusts
intensities; distortion correction adjusts the positions at which those
intensities are sampled. Both corrections are needed in this example.

.. _distortion-dataset: https://zenodo.org/records/3339629

Prepare the data
++++++++++++++++

Download the following files from the `Zenodo dataset <distortion-dataset_>`_
and save them together in a local directory named ``dist_corr``:

* ``tomographic_projections.hdf`` (19.9 GB): the projection images;
* ``darks.hdf`` (221.2 MB): the dark-field images; and
* ``flats.hdf`` (553.0 MB): the flat-field images.

The record also provides ``metadata.nxs``,
``dot_pattern.zip`` and ``reconstruction.zip``. These additional files are
not required by the pipeline below, which supplies rotation angles and
distortion coefficients directly in its YAML configuration.

Each HDF file stores its images at ``/entry/data/data``. The loader therefore
uses explicit dataset paths, with darks and flats in
:ref:`separate files <darks_flats>`. Rotation angles are supplied explicitly
from 0 to 180 degrees for 1801 projections.

GPU reconstruction pipeline
+++++++++++++++++++++++++++

This pipeline requires a CUDA-enabled GPU and the HTTomolibGPU, ToMoBAR and
HTTomolib backends. See :ref:`backends_list` and :ref:`howto_run` for setup
and execution options.

Download :download:`distortion_correction_data.yaml
<../../../pipelines_full/distortion_correction_data.yaml>` into ``dist_corr``,
or copy the pipeline below into a file with that name.

.. dropdown:: GPU pipeline for the distortion-correction dataset

   .. literalinclude:: ../../../pipelines_full/distortion_correction_data.yaml
      :language: yaml

The pipeline loads detector rows ``1000:1050`` to reconstruct a preview of
50 slices, retaining the full detector width and all projection angles.
The processing order is:

#. Correct the projections using the dark and flat fields.
#. Apply ``distortion_correction_proj_discorpy`` with the supplied calibration.
#. Reduce stripes that would produce ring artefacts.
#. Estimate the centre of rotation using ``find_center_vo``.
#. Apply ``minus_log`` and reconstruct with ``LPRec3d_tomobar``.
#. Save the reconstructed slices as TIFF images.

.. note::

   Calibration is performed separately, for example using Discorpy. These
   coefficients belong to this detector configuration; use the appropriate
   calibration for other data. A narrow vertical preview can limit the source
   rows available for interpolation near its boundaries. Use a wider preview
   or the full detector height when assessing the final reconstruction.

Dataset and method references
+++++++++++++++++++++++++++++

When using these data, cite the dataset: Vo, N. T., Atwood, R. C. and
Drakopoulos, M. (2019). *Tomographic data for demonstrating distortion
correction methods*, version v1. Zenodo.
`DOI: 10.5281/zenodo.3339629 <https://doi.org/10.5281/zenodo.3339629>`_.

The associated paper describes the preprocessing methods: Vo, N. T.,
Atwood, R. C. and Drakopoulos, M. (2019). *Preprocessing techniques for
removing artifacts in synchrotron-based tomographic images*.
Proceedings of SPIE 11113, 111131I.
`DOI: 10.1117/12.2530324 <https://doi.org/10.1117/12.2530324>`_.
