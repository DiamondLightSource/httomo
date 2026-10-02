.. _glossary:

Glossary
========

.. glossary::

   backend
      A scientific processing library whose functions HTTomo can call, such as
      HTTomolibGPU, HTTomolib or TomoPy. See :ref:`backends_list`; developers
      should also read :ref:`developers_httomo_backends`.

   block
      The memory-sized portion of one process's chunk passed through every
      method in a section before the next block is processed. See
      :ref:`blocks_data`.

   chunk
      The portion of a section's dataset assigned to one MPI process. See
      :ref:`chunks_data`.

   centre of rotation
      The detector coordinate corresponding to the sample's rotation axis.
      Accurate centring prevents characteristic reconstruction artefacts. See
      :ref:`centering`.

   image key
      A one-dimensional dataset that identifies each input frame as a
      projection, flat or dark image. See :ref:`darks_flats` and
      :ref:`create_nxtomo`.

   intermediate dataset
      A complete volume saved after a pipeline method, usually as an HDF5 file
      with the main array at ``/data``. See :ref:`info_logger`.

   loader
      The first pipeline method, responsible for locating input arrays and
      presenting projection data and auxiliary information to HTTomo. See
      :ref:`standard_tomo_loader`.

   method
      One loader, processing or output operation configured as an entry in a
      pipeline. See :ref:`pipeline_file_reference` for the fields that define
      a method entry and :ref:`reference_templates` for supported methods.

   method template
      A YAML description of a supported method, its parameters and defaults,
      generated from ``httomo-backends`` metadata. See
      :ref:`explanation_templates` and :ref:`reference_templates`.

   monitor
      Optional runtime instrumentation that reports aggregate or block-level
      timings. See :ref:`info_logger` and :ref:`run-httomo-indepth`.

   NXtomo
      The NeXus application definition for tomography data. An NXtomo entry
      links projection, image-key and rotation-angle datasets in a standard
      hierarchy that HTTomo can discover automatically. See
      :ref:`create_nxtomo`.

   padding
      Extra neighbouring slices supplied to a method so that operations near a
      block boundary have sufficient context. See :ref:`padding`.

   parameter sweep
      Repeated execution of a method for several candidate parameter values,
      with images saved for comparison. See :ref:`parameter_sweeping`.

   pattern
      The orientation in which methods consume data, principally projection or
      sinogram order. Pattern changes determine section boundaries and can
      require a re-slice; see :ref:`info_sections` and :ref:`info_reslice`.

   pipeline
      The ordered sequence of methods that HTTomo executes. Older material may
      call this a *process list*. See :ref:`explanation_process_list`,
      :ref:`pipeline_file_reference` and :ref:`tutorials_pl_templates`.

   preview
      A loader selection that crops the angular or detector dimensions before
      processing. See :ref:`previewing`.

   rank
      The identifier of one MPI process participating in a parallel run. Each
      rank normally receives one chunk; see :ref:`chunks_data` and
      :ref:`fig_execution_model`.

   re-slice
      Redistribution or transposition of data when consecutive sections use
      different processing patterns. See :ref:`info_reslice`.

   section
      A consecutive group of compatible methods that use the same processing
      pattern and are sized and executed together. See :ref:`info_sections`
      and :ref:`fig_execution_model`.

   side output
      A named value produced alongside the main dataset and referenced by a
      later pipeline method. See :ref:`side_output` for syntax and examples.

   wrapper
      HTTomo's adapter between a backend function and the common pipeline
      execution interface. See :ref:`info_wrappers` and
      :ref:`developer_architecture`.
