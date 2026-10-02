.. _real-data-lorentz:

Lorentz data
============

This example uses raw data from the `TomoBank`_ archive.

.. list-table::

   * - .. figure:: ../../../_static/real_data/sino_tomo088.jpg
          :width: 70%
          :align: center

          Dark/flat-field-corrected sinogram of the `Lorentz data set`_.

     - .. figure:: ../../../_static/real_data/recon_tomo088.jpg
          :width: 70%
          :align: center

          Reconstructed slice using the FBP method.

Download the `Lorentz data set`_. It is hosted using the Globus file management
system, which requires authentication. You can sign in using GitHub
credentials.

.. _TomoBank: https://tomobank.readthedocs.io/en/latest/

.. _Lorentz data set: https://tomobank.readthedocs.io/en/latest/source/data/docs.data.lorentz.html

After downloading the dataset, confirm that ``tomo_00088.h5`` is available,
then run one of the pipelines below.

TomoPy (CPU) pipeline
+++++++++++++++++++++

This pipeline uses TomoPy on the CPU, so TomoPy must be installed. See
:ref:`backends_list`. Copy the pipeline into a YAML file and
:ref:`run HTTomo <howto_run>`.

.. dropdown:: Standard 180 degrees pipeline using TomoPy (CPU) for tomo_00088.h5 dataset

    .. literalinclude:: ../../../pipelines_full/tomopy_tomobank.yaml
        :language: yaml

GPU pipeline
++++++++++++

If a CUDA-enabled GPU is available, the same dataset can be processed using
GPU-accelerated libraries. This can significantly reduce processing time for
suitable pipelines. Run the pipeline below in the same way as the CPU example.

.. dropdown:: GPU-enabled processing for tomo_00088.h5 dataset

    .. literalinclude:: ../../../pipelines_full/FBP3d_tomobar_tomobank.yaml
        :language: yaml
