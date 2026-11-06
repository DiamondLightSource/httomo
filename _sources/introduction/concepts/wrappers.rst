.. _info_wrappers:

Method wrappers
===============

HTTomo uses processing methods from external :ref:`backend libraries
<backends_list>`. Because these methods accept different inputs and produce
different outputs, HTTomo uses *wrappers* to provide a consistent interface between
the pipeline steps and each method action.

Wrappers prepare method inputs, handle CPU/GPU data transfers, invoke the backend
method and manage its outputs. Supporting information from :ref:`developers_httomo_backends`
describes requirements such as the data pattern, implementation, memory usage and
padding.

.. _fig_wrappers:

.. figure:: ../../_static/wrappers/httomo_wrappers.png
   :alt: HTTomo wrapper layer and its five main wrapper types
   :align: center
   :width: 100%

   The HTTomo wrapper layer connects backend methods to the processing pipeline and
   handles their method-specific inputs and outputs.

Wrapper types
~~~~~~~~~~~~~

HTTomo provides five main wrapper types:

- **Generic:** passes data to standard processing methods and returns the result.
- **Normalisation:** supplies projection data, flats and darks, then returns
  normalised data.
- **Centering:** prepares the required projections or sinograms and produces a
  centre-of-rotation value.
- **Reconstruction:** supplies the data, projection angles and reconstruction
  parameters, including the centre of rotation.
- **Image saver:** transfers data to the CPU when necessary and writes images to
  storage without changing the pipeline data.