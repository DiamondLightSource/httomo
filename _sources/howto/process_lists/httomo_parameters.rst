.. _howto_proc_httomo_params:

==========================
HTTomo-specific parameters
==========================

Method Parameters vs. HTTomo Method Parameters
----------------------------------------------

In addition to parameters for a method that influence the processing that the
method performs, there are some parameters that can be used in a method's YAML
configuration that modify in some way how HTTomo handles the method (for
example, whether or not to write the output of the method to a file).

Since these parameters are *not* parameters for the method itself and are
instead specific to HTTomo, the sections below demonstrate these extra
parameters, explaining what they are for and giving examples of how they can be
used.

.. toctree::
   :maxdepth: 1

   side_outputs/side_out
   save_results/save_results
