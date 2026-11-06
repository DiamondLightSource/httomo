.. _developers_add_own_method:

Adding a processing method
**************************

Adding a method spans a processing library, ``httomo-backends`` and HTTomo.
Use the following workflow to keep each part in the correct project:

.. figure:: ../_static/add_method_workflow.svg
   :align: center
   :alt: Four-step workflow for implementing, registering and validating a new HTTomo processing method
   :width: 100%

   Implement and test the function first; describe it for HTTomo only after its
   interface is stable.

HTTomo usually exposes processing functions supplied by a separate scientific
library. Functions from modular packages can often be integrated directly when
they accept an array and explicit processing parameters. Functions with more
complex interfaces may require an HTTomo :ref:`wrapper <info_wrappers>`.

After integrating and testing the function in its backend library, register it
in ``httomo-backends``. That package supplies the execution metadata HTTomo
needs for sectioning and memory management, and generates the user-facing YAML
template. Follow :ref:`developers_httomo_backends` for the complete workflow.
