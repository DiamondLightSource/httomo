.. _side_output:

Side outputs
++++++++++++

Methods normally pass their processed dataset to the next method. Some methods
also produce supplementary values, called *side outputs*, which can be used as
parameters by methods later in the pipeline. A common example is passing a
calculated centre of rotation to a reconstruction method.

.. figure:: ../../_static/side_output_reference.svg
   :width: 100%
   :alt: Main dataset and side-output flows between two pipeline methods

   The processed dataset follows the main pipeline, while the named side output
   is passed to a later method parameter.

Define and reference a side output
##################################

The producing method requires:

* a unique ``id``;
* a ``side_outputs`` mapping from the method's output name to a pipeline name.

The consuming method refers to the value using
``${{id.side_outputs.name}}``:

.. code-block:: yaml
   :emphasize-lines: 11-13,18

   - method: find_center_vo
     module_path: httomolibgpu.recon.rotation
     parameters:
       ind: null
       smin: -50
       smax: 50
       srad: 6.0
       step: 0.25
       ratio: 0.5
       drop: 20
     id: centering
     side_outputs:
       cor: centre_of_rotation

   - method: FBP3d_tomobar
     module_path: httomolibgpu.recon.algorithm
     parameters:
       center: ${{centering.side_outputs.centre_of_rotation}}
       filter_freq_cutoff: 1.0
       recon_size: null
       recon_mask_radius: 0.95

The referenced method must appear *before* the method that uses its output, and
each explicit ``id`` must be *unique* within the pipeline.

.. note::

   Side outputs and their references are generated automatically by the
   `YAML generator
   <https://diamondlightsource.github.io/httomo-backends/utilities/yaml_generator.html>`_.
   They normally do not need to be changed when adapting a generated pipeline.
