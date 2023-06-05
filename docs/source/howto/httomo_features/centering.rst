.. _centering:

Sinogram centering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatic calculation OR manual input of the centre of rotation are possible in Httomo.


Auto-centering
====================================================

RE-WRITE!

The auto-centering plugin (VoCentering) can be added to a process list before the reconstruction
plugin.  The value calculated in the centering routine is automatically passed to the reconstruction
and will override the centre_of_rotation parameter in the reconstruction plugin. The auto-centering
plugin is computationally expensive and should only be applied to previewed data.  There are two ways
to achieve this:

1. Apply previewing in the loader plugin to reduce the size of the processed data.

and/or

2. Apply previewing in VoCentering plugin (this will not reduce the size of the data).

.. note:: If you have applied previewing in the loader and again in the centering plugin you will be
          applying previewing to the previewed (reduced size) data.

See :ref:`autocentering`


Manual Centering
======================================================

RE-WRITE!

Ensure the VoCentering algorithm is not in the process list (remove or comment it).  Modify the centre_of_rotation value in the reconstruction plugin, see
:ref:`manualcentering`.  If the manual centering value is approximate you can apply parameter
tuning, see :ref:`cor_parameter_tuning`


