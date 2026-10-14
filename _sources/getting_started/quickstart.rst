.. _quickstart:

10-minute quickstart
====================

This example downloads HTTomo's small standard test dataset, validates a CPU
pipeline and reconstructs the data with TomoPy. It does not require a GPU.

Prerequisites
-------------

Install HTTomo by following :ref:`installation_cpu_only`. The example pipeline uses
TomoPy and HTTomolib.


Download the data and pipeline
------------------------------

Create an empty working directory and download the test dataset and CPU
pipeline from the current HTTomo version:

.. code-block:: console

   $ mkdir httomo-quickstart
   $ cd httomo-quickstart
   $ curl -L -O \
       https://raw.githubusercontent.com/DiamondLightSource/httomo/main/tests/test_data/tomo_standard.nxs
   $ curl -L -O \
       https://raw.githubusercontent.com/DiamondLightSource/httomo/main/docs/source/pipelines_full/tomopy_gridrec.yaml

The `tomo_standard.nxs test dataset
<https://github.com/DiamondLightSource/httomo/blob/main/tests/test_data/tomo_standard.nxs>`_
is approximately 9 MB. It contains 180 projections plus flat and dark fields,
with detector frames of 128 by 160 pixels. ``tomopy_gridrec.yaml`` performs
normalisation, automatic centre finding, gridrec reconstruction, rescaling and
TIFF output.

Validate the pipeline
---------------------

Check both the pipeline syntax and the dataset paths before processing:

.. code-block:: console

   $ python -m httomo check tomopy_gridrec.yaml tomo_standard.nxs

A successful check finishes without validation errors.

Run the reconstruction
----------------------

Create the parent output directory, then run HTTomo:

.. code-block:: console

   $ mkdir output
   $ python -m httomo run \
       tomo_standard.nxs tomopy_gridrec.yaml output

HTTomo creates a timestamped directory below ``output``. The run reconstructs
128 slices of 160 by 160 pixels and writes them to the ``images8bit_tif``
subdirectory as TIFF files. The run directory also contains an intermediate HDF5
reconstruction, ``tomopy_gridrec.yaml``, ``user.log`` and ``debug.log``.

Open one of the TIFF files to confirm that the reconstruction completed. The
centre reported in ``user.log`` should be close to 79.5 pixels.

Next steps
----------

* Use :ref:`howto_run` with your own input data.
* Consult :ref:`troubleshooting` if validation or execution fails.
* Check :ref:`versioned_downloads` when using another HTTomo release.
