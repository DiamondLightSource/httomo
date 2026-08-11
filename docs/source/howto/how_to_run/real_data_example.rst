.. _real-data-example:

Real data processing
====================

This section presents an example of processing real experimental data using HTTomo.

Before starting, we assume that HTTomo has been successfully installed. If you have not installed HTTomo yet, please follow the
:ref:`installation_main`.

We also recommend running the :ref:`run_tests` to verify that all required dependencies are installed and that the framework is 
functioning correctly.

For this example, we will use raw data from the `TomoBank`_ data archive. Please download the `Lorentz data set`_. The dataset is 
hosted using the Globus file management system, which requires authentication. You can sign in using your GitHub credentials.

.. _TomoBank: https://tomobank.readthedocs.io/en/latest/

.. _Lorentz data set: https://tomobank.readthedocs.io/en/latest/source/data/docs.data.lorentz.html

Once the dataset has been downloaded, you should have the file :code:`tomo_00088.h5` on your disk. You can then proceed with running a simple HTTomo pipeline.

TomoPy (CPU) pipeline
+++++++++++++++++++++

This pipeline uses the CPU implementation of the TomoPy library. TomoPy must be installed before running the pipeline. 
See :ref:`backends_list`.

Running this pipeline requires TomoPy package to be installed, see :ref:`backends_list`. Copy the following pipeline into a YAML file,
for example: :code:`tomopy_tomo_00088.yaml`. 

.. dropdown:: Standard 180 degrees pipeline using TomoPy (CPU) for tomo_00088.h5 dataset

    .. literalinclude:: ../../pipelines_full/tomopy_tomobank.yaml
        :language: yaml


Then run HTTomo according to :ref:`howto_run_outside_diamond` documentation. In particular, provide the path to the input dataset, the pipeline YAML file, and the output directory.

.. code-block:: console

    $ python -m httomo run path/to/tomo_00088.h5 tomopy_tomo_00088.yaml /path/to/output_folder

GPU pipeline
++++++++++++

If a CUDA-enabled GPU is available, the same dataset can be processed using GPU-accelerated libraries. 
This can significantly reduce the processing time for suitable pipelines. Run the pipeline bellow in a similar way as explained above. 

.. dropdown:: GPU-enabled processing for tomo_00088.h5 dataset

    .. literalinclude:: ../../pipelines_full/FBP3d_tomobar_tomobank.yaml
        :language: yaml