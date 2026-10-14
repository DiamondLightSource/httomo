.. _tutorials_pl_templates:

Ready-to-use pipelines
======================

These complete HTTomo pipelines are starting points for common workflows.
Select one using :ref:`choose_pipeline`, then adapt its loader and method
parameters to the input data. See :ref:`explanation_process_list` for the
underlying concepts and :ref:`howto_process_list` for configuration guidance.

.. warning::

   These examples track the current HTTomo development version. For a tagged
   release, use the matching pipeline from :ref:`versioned_downloads`.

.. _tutorials_pl_templates_gpu:

GPU pipelines
-------------

These pipelines combine GPU methods from HTTomolibGPU with CPU output methods
from HTTomolib. Reconstruction methods also require TomoBAR. See
:ref:`backends_list` for the role of each library.

.. dropdown:: FBP3d with find_center_vo centring and image output

   .. literalinclude:: ../pipelines_full/FBP3d_tomobar.yaml
      :language: yaml

.. dropdown:: FBP3d with phase-correlation centring and downsampling

   .. literalinclude:: ../pipelines_full/titaren_center_pc_FBP3d_resample.yaml
      :language: yaml

.. dropdown:: LPRec3d reconstruction

   .. literalinclude:: ../pipelines_full/LPRec3d_tomobar.yaml
      :language: yaml

.. dropdown:: FBP3d followed by total-variation denoising

   .. literalinclude:: ../pipelines_full/FBP3d_tomobar_denoising.yaml
      :language: yaml

.. dropdown:: FISTA3d with total-variation regularisation

   This iterative example is intended for noisy or undersampled data.

   .. literalinclude:: ../pipelines_full/FISTA3d_tomobar.yaml
      :language: yaml

.. _tutorials_pipelines:

Tutorial pipelines
------------------

These pipelines accompany :ref:`data_tutorials`, where the input data is
provided or generated.

.. dropdown:: TomoPy CPU pipeline for the Lorentz dataset

   .. literalinclude:: ../pipelines_full/tomopy_tomobank.yaml
      :language: yaml

.. dropdown:: GPU pipeline for the Lorentz dataset

   .. literalinclude:: ../pipelines_full/FBP3d_tomobar_tomobank.yaml
      :language: yaml

.. _tutorials_pl_templates_dls:

Diamond-specific pipelines
--------------------------

These examples implement processing strategies used at Diamond Light Source.
Required parameters and calibration files can differ between beamlines.

.. dropdown:: Convert a 360° scan to 180°, apply Paganin filtering and reconstruct

   .. literalinclude:: ../pipelines_full/deg360_paganin_FBP3d_tomobar.yaml
      :language: yaml

.. dropdown:: Correct distortion, convert a 360° scan and reconstruct

   .. literalinclude:: ../pipelines_full/deg360_distortion_FBP3d_tomobar.yaml
      :language: yaml

.. _tutorials_pl_templates_sweeps:

Parameter-sweep pipelines
-------------------------

Sweep runs automatically add image output after each swept method. Do not add a
separate image-saving method there. See :ref:`parameter_sweeping` for syntax
and execution details.

.. dropdown:: Sweep centre-of-rotation values with !SweepRange

   .. literalinclude:: ../pipelines_full/sweep_center_FBP3d_tomobar.yaml
      :language: yaml
      :emphasize-lines: 36-39

.. dropdown:: Sweep selected Paganin ratio values with !Sweep

   .. literalinclude:: ../pipelines_full/sweep_paganin_FBP3d_tomobar.yaml
      :language: yaml
      :emphasize-lines: 51-54

.. _tutorials_pl_templates_cpu:

CPU pipeline
------------

Use the TomoPy example when a CUDA-capable GPU is unavailable. Performance
depends on the input, method parameters, CPU resources and process count.

.. dropdown:: TomoPy gridrec with automatic centring

   .. literalinclude:: ../pipelines_full/tomopy_gridrec.yaml
      :language: yaml
