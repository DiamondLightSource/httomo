.. _real-data-lorentz:

Lorentz data
============

For this example, we will use raw data from the `TomoBank`_ data archive. 

.. list-table::

   * - .. figure:: ../../../_static/real_data/sino_tomo088.jpg
          :width: 70%
          :align: center

          Dark/flat-field-corrected sinogram of the `Lorentz data set`_.

     - .. figure:: ../../../_static/real_data/recon_tomo088.jpg
          :width: 70%
          :align: center

          Reconstructed slice using the FBP method.

Please download the `Lorentz data set`_. The dataset is 
hosted using the Globus file management system, which requires authentication. You can sign in using your GitHub credentials.

.. _TomoBank: https://tomobank.readthedocs.io/en/latest/

.. _Lorentz data set: https://tomobank.readthedocs.io/en/latest/source/data/docs.data.lorentz.html

Once the dataset has been downloaded, you should have the file :code:`tomo_00088.h5` on your disk. You can then proceed with running a simple HTTomo pipeline.

TomoPy (CPU) pipeline
+++++++++++++++++++++

This pipeline uses the CPU implementation of the TomoPy library. TomoPy must be installed before running the pipeline. 
See :ref:`backends_list`.

Running this pipeline requires TomoPy package to be installed, see :ref:`backends_list`. Copy the following pipeline into a YAML file and :ref:`run HTTomo <howto_run>`.

.. dropdown:: Standard 180 degrees pipeline using TomoPy (CPU) for tomo_00088.h5 dataset

    .. literalinclude:: ../../../pipelines_full/tomopy_tomobank.yaml
        :language: yaml

GPU pipeline
++++++++++++

If a CUDA-enabled GPU is available, the same dataset can be processed using GPU-accelerated libraries. 
This can significantly reduce the processing time for suitable pipelines. Run the pipeline bellow in a similar way as explained above. 

.. dropdown:: GPU-enabled processing for tomo_00088.h5 dataset

    .. literalinclude:: ../../../pipelines_full/FBP3d_tomobar_tomobank.yaml
        :language: yaml
