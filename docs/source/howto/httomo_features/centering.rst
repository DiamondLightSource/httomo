.. _centering:

Centre of rotation (CoR)
^^^^^^^^^^^^^^^^^^^^^^^^^
An important procedure to ensure the correctness of reconstruction is to find the CoR of a sinogram. 

Automatic calculation OR manual input of the centre of rotation are possible in Httomo.


.. _centering_auto:

Auto-centering
===============

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

.. _centering_manual:

Manual Centering
=================
In case when :ref:`centering_auto` does not work (e.g. the data is corrupted, incomplete or/and not within the field of view), 
one can use the manual centering with :ref:`parameter_tuning`.

For manual centering you need to do the following steps:

1. Ensure that the auto centering estimation method is not in the process list (remove or comment it). 
2. Modify the centre of rotation value :code:`center` in the reconstruction plugin by substituting a number.
3. If you would like to sweep across multiple CoR values, you can do that with a special phrase in your template :code:`!Sweep` or ::code:`!SweepRange`. Please see more on :ref:`parameter_tuning_range`.



