.. _reference_templates:

Available methods
=================

HTTomo processing methods are provided by the :ref:`backends_list`. Each
available method has a YAML template containing its module path, parameters
and default values. Copy these templates when :ref:`configuring a pipeline
<how_to_configure_pipeline>`.

The generated references below track the latest HTTomo development version.
For a tagged HTTomo release, download the matching templates from
:ref:`versioned_downloads`.

.. _latest_templates:

Current method reference
------------------------

The method templates are generated and published by `HTTomo-backends
<https://diamondlightsource.github.io/httomo-backends/>`_. Select the library
used by your pipeline:

* `HTTomolibgpu methods (GPU)
  <https://diamondlightsource.github.io/httomo-backends/backends/templates.html#httomolibgpu-modules>`_
* `HTTomolib methods (CPU)
  <https://diamondlightsource.github.io/httomo-backends/backends/templates.html#httomolib-modules>`_
* `TomoPy methods (CPU)
  <https://diamondlightsource.github.io/httomo-backends/backends/templates.html#tomopy-modules>`_

.. note::
   At Diamond Light Source, the current references correspond to the
   :code:`httomo/latest` module. Use :ref:`versioned_downloads` for production
   runs tied to a particular release.
