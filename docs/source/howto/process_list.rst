Configure process list using templates
----------------------------------------------------

This section explains how to build a `process list <https://diamondlightsource.github.io/httomo/explanation/process_list.html>`_ from YAML `templates <https://diamondlightsource.github.io/httomo/reference/templates.html>`_.
We focus on several important elements which are helpful to keep in mind, before configuring the process list. The order of methods in the pipeline, the union of methods and reslicing operations.
The better understanding of this will enable you to build more computationally efficient pipelines. To avoid errors during the run, please do not forget validating the final process list with the `YAML checker <https://diamondlightsource.github.io/httomo/utilities/yaml_checker.html>`_ tool.

Before start, please become familiar with the `YAML format <https://diamondlightsource.github.io/httomo/explanation/yaml.html>`_ and use editors that support it. We can recommend Visual Studio Code, Atom, Notepad++.


Methods order
==================
To build a process list you will need to copy-paste the content of YAML files from the provided `templates <https://diamondlightsource.github.io/httomo/reference/templates.html>`_.
The general rules are the following: 

* Any process list starts with a loader which is `provided <https://diamondlightsource.github.io/httomo/api/httomo.data.hdf.loaders.html>`_ by HTTomo.
* The execution order of the methods in the process list is from the top to the bottom and **sequential**.

For example, for tomographic processing we can build the following process list by using TomoPy `templates <https://diamondlightsource.github.io/httomo/reference/templates.html>`_ 
and HTTomo loader to read `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ dataset.

.. code-block:: yaml

  httomo.data.hdf.loaders:
      standard_tomo:
        name: tomo
        data_path: entry1/tomo_entry/data/data
        image_key_path: entry1/tomo_entry/instrument/detector/image_key
        dimension: 1
        preview:
          - 
          - 
          - 
        pad: 0
  tomopy.prep.normalize:
      normalize:
        data_in: tomo
        data_out: tomo
        cutoff: null
  tomopy.prep.normalize:
      minus_log:
        data_in: tomo
        data_out: tomo
  tomopy.prep.stripe:
      remove_stripe_fw:
        data_in: tomo
        data_out: tomo
        level: null
        wname: db5
        sigma: 2
        pad: true
  tomopy.recon.rotation:
      find_center_vo:
        data_in: tomo
        data_out: cor
        ind: mid
        smin: -50
        smax: 50
        srad: 6
        step: 0.25
        ratio: 0.5
        drop: 20
  tomopy.prep.stripe:
      remove_stripe_fw:
        data_in: tomo
        data_out: tomo
        level: null
        wname: db5
        sigma: 2
        pad: true        
  tomopy.recon.algorithm:
      recon:
        data_in: tomo
        data_out: tomo
        center: cor
        sinogram_order: false
        algorithm: gridrec
        init_recon: null
        #additional parameters': AVAILABLE
  tomopy.misc.corr:
      median_filter:
        data_in: tomo
        data_out: tomo
        size: 3
        axis: 0


Methods unions
==================


Reslicing
==================