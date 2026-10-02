.. default-role:: math
.. _centering:

Centre of Rotation
^^^^^^^^^^^^^^^^^^

The Centre of Rotation (CoR) aligns the sample's rotation axis with the
acquisition coordinate system, as shown in :numref:`fig_centerscheme`.
Reconstruction assumes this alignment, so an inaccurate CoR can distort the
result and invalidate later analysis.

An offset sinogram (`d` in :numref:`fig_centerscheme`) produces arching
artefacts around object boundaries, as shown in :numref:`fig_center_find`.
These artefacts increase with the distance from the correct CoR (`d=0`), so
CoR estimation typically searches for the value that minimises them.

.. _fig_centerscheme:
.. figure::  ../../_static/cor/CoR.svg
    :scale: 55 %
    :alt: CoR scheme for tomography

    The CoR offset `d` translates the sample coordinates `(x,y)` into the
    acquisition coordinates `(s,p)`: `(s = x +- d, p = y)`.

.. _fig_center_find:
.. figure::  ../../_static/cor/corr_select.png
    :scale: 85 %
    :alt: Finding CoR

    Reconstructions with different CoR values. Boundary artefacts decrease as
    `d` approaches the correct value.

CoR in HTTomo
=============

Every reconstruction template provides a :code:`center` parameter. Set it
automatically (see :ref:`centering_auto`) or manually (see
:ref:`centering_manual`).

.. _centering_auto:

Auto-centering
===============

Several methods can estimate the CoR automatically. DLS commonly uses Nghia
Vo's Fourier-based sinogram method (`paper`_), implemented by TomoPy and
HTTomolibGPU. In HTTomo it is available as the ``find_center_vo`` template;
see :ref:`reference_templates`. If one automatic method fails, try another
`HTTomolibGPU centring method`_.

.. _paper: https://doi.org/10.1364/OE.22.019078
.. _HTTomolibGPU centring method: https://diamondlightsource.github.io/httomolibgpu/api/httomolibgpu.recon.rotation.html

To use automatic centering:

1. Add the centering method before reconstruction, preferably immediately
   after the loader; see :ref:`pl_conf_order`.
2. Store its calculated CoR as a :ref:`side_output`.
3. Reference that output in the reconstruction method's :code:`center`
   parameter.

.. code-block:: yaml
  :emphasize-lines: 13,17

  - method: find_center_vo
    module_path: httomolibgpu.recon.rotation
    parameters:
      ind: mid
      smin: -50
      smax: 50
      srad: 6
      step: 0.25
      ratio: 0.5
      drop: 20
    id: centering
    side_outputs:
      cor: centre_of_rotation
  - method: FBP
    module_path: httomolibgpu.recon.algorithm
    parameters:
      center: ${{centering.side_outputs.centre_of_rotation}}
      filter_freq_cutoff: 1.1
      recon_size: null
      recon_mask_radius: null


.. _centering_manual:

Manual Centering
=================

Automatic centering can fail when projection data is corrupt or incomplete,
or when the sample extends beyond the detector's field of view. In these cases,
set the CoR manually. :ref:`parameter_sweeping` can help identify the value.

To set it without parameter sweeping:

1. Remove or comment out the automatic centering method.
2. Replace the side-output reference in the reconstruction method's
   :code:`center` parameter with a numeric value.
