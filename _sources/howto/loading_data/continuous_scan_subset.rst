.. _continuous_scan_subset_selection:

Continuous-scan subsets
^^^^^^^^^^^^^^^^^^^^^^^

A single 3D HDF5 dataset can contain several tomography scans arranged along
the angular dimension. Use :code:`continuous_scan_subset` to load one scan by
specifying its start and stop indices. As with Python slicing, the stop index is
excluded.

This example selects indices 90 through 179:

.. literalinclude:: ../../../../tests/samples/pipeline_template_examples/testing/loader_with_offset_param.yaml
   :language: yaml
   :emphasize-lines: 7-9

The option can be combined with :ref:`previewing` to crop the detector
dimensions and with :ref:`darks_flats` to load external darks or flats.
Its start and stop values replace ``preview.angles``. If
``--continuous-scan-subset`` is also supplied on the command line, the
command-line values replace the values in the pipeline.
