.. _real-data-distortion:

Distortion-correction data
==========================

This example follows `Exercise 13 of the HTTomo workshop`_. It demonstrates
how to apply a supplied optical-distortion calibration to projection images
before reconstructing a 180-degree scan.

Detector optics can displace image features and introduce shape errors and
streak artefacts in a reconstruction. Dark/flat-field correction adjusts
intensities; distortion correction adjusts the positions at which those
intensities are sampled. Both corrections are needed in this example.

.. _Exercise 13 of the HTTomo workshop: https://dkazanc.github.io/httomo-workshop26/#exercise13

Prepare the data
++++++++++++++++

Use the ``dist_corr`` folder supplied for the workshop. At Diamond, the
workshop data are under
``/dls/k11/data/2026/mg47440-2/processing/httomo-workshop/data``.
Confirm that the following files are available in ``dist_corr``:

* ``tomographic_projections.hdf``: the 1801 projection images;
* ``darks.hdf``: the dark-field images; and
* ``flats.hdf``: the flat-field images.

Each file stores its images at ``/entry/data/data``. The loader therefore
uses explicit dataset paths, with darks and flats in
:ref:`separate files <darks_flats>`. Rotation angles are supplied explicitly
from 0 to 180 degrees for 1801 projections.

.. note::

   These commands assume access to the workshop data. If you do not have
   access to the Diamond visit directory, obtain the three files from the
   workshop organisers and keep them together in a local ``dist_corr`` folder.

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

The distortion centre (``xcenter``, ``ycenter``) is expressed in pixels in
the original detector coordinates. It is different from the centre of
rotation used for reconstruction. Keep the supplied centre and ``list_fact``
coefficients unchanged for this dataset: HTTomo passes the loader's preview
offsets to the distortion method automatically. ``order: 1`` selects linear
interpolation and ``mode: reflect`` selects the interpolation boundary mode.
``metadata_path: null`` tells HTTomo to use the coefficients given directly
in the YAML rather than read them from a calibration file.

.. note::

   Calibration is performed separately, for example using Discorpy. These
   coefficients belong to this detector configuration; use the appropriate
   calibration for other data. A narrow vertical preview can limit the source
   rows available for interpolation near its boundaries. Use a wider preview
   or the full detector height when assessing the final reconstruction.

Run HTTomo
++++++++++

From the ``dist_corr`` directory, run the downloaded pipeline. At Diamond,
load the latest HTTomo module and use the cluster wrapper described in
:ref:`howto_run_at_diamond`:

.. code-block:: bash

   module load httomo/latest
   httomo_mpi tomographic_projections.hdf distortion_correction_data.yaml \
     output_distortion_corrected

With a local HTTomo installation, use:

.. code-block:: bash

   httomo run tomographic_projections.hdf distortion_correction_data.yaml \
     output_distortion_corrected

The relative paths ``darks.hdf`` and ``flats.hdf`` are resolved from the
directory where you run the command. If you run from elsewhere, replace
these paths in the YAML with absolute paths.

Compare with an uncorrected reconstruction
++++++++++++++++++++++++++++++++++++++++++

Copy the YAML to ``distortion_uncorrected.yaml`` and remove the entire
``distortion_correction_proj_discorpy`` method block. Keep all other stages,
parameters and preview bounds the same, including the centre finder, so
each run estimates its centre of rotation from its own processed data.
Run this pipeline into a separate output directory:

.. code-block:: bash

   httomo_mpi tomographic_projections.hdf distortion_uncorrected.yaml \
     output_distortion_uncorrected

For a local installation, replace ``httomo_mpi`` with ``httomo run``.
Open corresponding TIFF slices from the ``images`` subfolder of each run
and use the same display range. Compare feature shapes, boundary sharpness
and streaking, especially towards the edge of the reconstruction. Correction
can shift features, so compare corresponding structures as well as the same
pixel coordinates.

After the preview comparison, increase the vertical range or remove
``preview`` to reconstruct the full volume. Keep the same preview in both
pipelines when repeating the comparison.
