.. default-role:: math
.. _centering:

Centre of Rotation (CoR)
^^^^^^^^^^^^^^^^^^^^^^^^

What is it?
===========
Identifying the optimal Centre of Rotation (CoR) parameter is an important 
procedure to ensure the correctness of the reconstruction. It is crucial to find it 
precisely, as the incorrect value can lead to distortions in reconstructions, therefore making post processing and
quantification incorrect.  

The required CoR parameter places the object (a scanned sample) into a coordinate system of the scanning device to ensure
that the object is centered and rotates around its axis in this system, see :numref:`fig_centerscheme`. This is essential for
valid reconstruction as the back projection model assumes a central placing of a sample with respect to one detector's axis
(usually horizontal for synchrotrons).

The CoR estimation problem is also sometimes referred to the centered sinogram. 
If the sinogram is not centered and there is an offset (`d` in :numref:`fig_centerscheme`), the reconstruction 
will result in strong arching artefacts at the boundaries of reconstructed objects, see :numref:`fig_center_find`. 
Further from the optimal CoR value (here `d=0`), one should expect more pronounced arching. 
Therefore the optimisation problem usually involves minimising the artefacts in the reconstructed images
by varying the CoR value.

.. _fig_centerscheme:
.. figure::  ../../_static/CoR.svg
    :scale: 55 %
    :alt: CoR scheme for tomography

    The CoR parameter can be defined as a distance `d` that translates the coordinate system `(x,y)` of the scanned object to the coordinate system `(s,p)` of the acquisition device. This leads to a simple linear mapping `(s = x + d, p = y)`.

.. _fig_center_find:
.. figure::  ../../_static/cor/corr_select.png
    :scale: 85 %
    :alt: Finding CoR

    The reconstructions using different CoR values. Incorrectly centered sinogram results in strong arching artefacts on the boundaries of the reconstructed object. Note how the arching is being reduced when closer to the correct value. 

CoR in HTTomo
=============

The CoR parameter is present in any reconstruction template as 
the :code:`center` parameter and can be configured either automatically (see :ref:`centering_auto`)
or manually (see :ref:`centering_manual`).

.. _centering_auto:

Auto-centering
===============

There is a `variety <https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=center+of+rotation+tomography&btnG=>`_ of 
methods to estimate CoR automatically. At DLS we use a centering method which 
was developed by Nghia Vo and it relies on Fourier analysis of a sinogram, see the `paper <https://opg.optica.org/directpdfaccess/a01b4da4-e3cf-474a-b428ca517599c609_297315/oe-22-16-19078.pdf?da=1&id=297315&seq=0&mobile=no>`_. 
This method is implemented in both TomoPy and HTTomolibgpu libraries and 
available for HTTomo as *find_center_vo* template, see :ref:`reference_templates`. 

Here are the steps to enable the auto-centering: 

1. The auto-centering method should be added to a process list before the reconstruction.
2. The value calculated in the centering routine :code:`cor` stored in the output dataset :code:`data_out`.
3. The CoR value :code:`cor` needs to be passed to the reconstructor into the :code:`center` parameter. 

.. code-block:: yaml
  :emphasize-lines: 4,16

   - tomopy.recon.rotation:
       find_center_vo:
         data_in: tomo
         data_out: cor
         ind: mid
         smin: -50
         smax: 50
         srad: 6.0
         step: 0.25
         ratio: 0.5
         drop: 20
   - tomopy.recon.algorithm:
       recon:
         data_in: tomo
         data_out: tomo
         center: cor
         sinogram_order: false
         algorithm: gridrec
         init_recon: null

.. note:: The auto-centering plugin is computationally expensive and therefore one can specify :code:`ind: mid` to have it calculated only for a middle slice of your previewed data, see :ref:`previewing`. 

.. _centering_manual:

Manual Centering
=================
In the case when :ref:`centering_auto` does not work (e.g. the data is corrupted, incomplete or/and not within the field of view of the detector),
one can use manual centering with :ref:`parameter_tuning` enabled.

For manual centering you need to do the following steps:

1. Ensure that the auto centering estimation method is not in the process list (remove or comment it). 
2. Modify the centre of rotation value :code:`center` in the reconstruction plugin by substituting a number instead of the :code:`cor` parameter.
3. If you would like to sweep across multiple CoR values, you can do that with a special phrase in your template :code:`!Sweep` or :code:`!SweepRange`. Please see more at :ref:`parameter_tuning_range`.



