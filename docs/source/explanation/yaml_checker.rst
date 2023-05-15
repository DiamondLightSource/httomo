Yaml Checker - Why use it?
--------------------------

First, what is YAML?
####################

.. code-block:: yaml

    # What does YAML mean?​
    YAML:​
        -Y: YAML​
        -A: Ain't​
        -M: Markup​
        -L: Language

.. note::

    Markup Language is language which annotates text using tags or keywords
    in order to define how content is displayed


* YAML documents are a collection of key-value pairs​
* Indentation is used to denote structure
* The length of the indentation should not matter, as long as you are consistent inside the same file

Format examples
===============

.. code-block:: yaml

    # A list of numbers using hyphens:​
    numbers:​
        - one​
        - two​
        - three​
    ​
    # The inline version:​
    numbers: [ one, two, three ]


* Indent with *spaces*, *not* tabs​.
* There must be spaces between elements​


.. code-block::

    # This is correct​
    name: tomo​

    # This will fail​
    name:tomo


.. code-block:: yaml

    # Strings don't require quotes:​
    title: Introduction to YAML​
    ​
    # But you can still use them if you prefer:​
    title-with-quotes: 'Introduction to YAML'​


Using a YAML file to specify a pipeline of functions for data processing is a common
and practical approach to creating a reproducible data analysis workflow. This approach
can help to ensure that your pipeline runs correctly and consistently over time,
regardless of the platform or environment in which it is executed. By using a YAML file
to define your pipeline, we are providing a user interface that is simple and intuitive for scientists
to use. This can be especially helpful for those who are less familiar with programming in general
or are new to the specific tools and libraries you are using.

We have a YAML checker that can help you validate your YAML file, before you use it to run your pipeline,
and we highly recommend that you use it. The checker will help you to identify any errors in your YAML file.

Usage
=====

.. code-block:: console

   $ python -m httomo check YAML_CONFIG IN_DATA


.. note::

    - Use this :code:`check` command before you use the :code:`run` command to run your pipeline.
    - The :code:`YAML_CONFIG` is the path to your YAML file and :code:`IN_DATA`` is the path to your input data.
    - :code:`IN_DATA` is optional, but if you provide it, the yaml checker will be checking that the paths
      to the data and keys in the :code:`YAML_CONFIG` file match the paths and keys in the input file (:code:`IN_DATA`).


For example, if you have the following as a :code:`YAML_CONFIG` file saved as :code:`config.yaml`:

.. code-block:: yaml

    - httomo.data.hdf.loaders:
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
    - tomopy.prep.normalize:
        normalize:
          data_in: tomo
          data_out: tomo
          cutoff: null
          averaging: mean
    - tomopy.prep.normalize:
        minus_log:
          data_in: tomo
          data_out: tomo
    - httomolib.misc.images:
        save_to_images:
          data_in: tomo
          subfolder_name: images
          axis: 0
          file_format: tif
          bits: 8
          perc_range_min: 0.0
          perc_range_max: 100.0
          jpeg_quality: 95


And you run the YAML checker with:

.. code-block:: console

   $ python -m httomo check config.yaml


You will get the following output:

.. code-block:: console

    Checking that YAML_CONFIG includes only one method from each module...

    Doing a sanity check first...
    Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...
    Sanity check of the YAML_CONFIG was successfully done...

    'One method per module' check was also successfully done...

    Checking that the first method in the pipeline is a loader...
    Loader check successful!!

    Checking 'standard_tomo' and its parameters...
    Checking 'normalize' and its parameters...
    Checking 'minus_log' and its parameters...
    Checking 'save_to_images' and its parameters...

    YAML validation successful!! Please feel free to use the `run` command to run the pipeline.


The Yaml check was successful here because your yaml file was properly indented and had valid mappings and tags.
It also included valid parameters for each method used from TomoPy, HTTomoLib, etc.

But if you had the following as a :code:`YAML_CONFIG` file saved as :code:`config.yaml`:

.. code-block:: yaml

    - httomo.data.hdf.loaders:
        standard_tomo:
          name: tomo
          data_path: entry1/tomo_entry/data/data
          image_key_path: entry1/tomo_entry/instrument/detector/image_key
          preview:
            - 
            - start: 30
              stop: 60
            - 
        pad: 0
    - tomopy.misc.corr:
        median_filters:
          data: tomo
          data_out: tomo
          size: tomo
          axis: 0

And then you run the YAML checker, you get:

.. code-block:: console

    $ python -m httomo check config.yaml
    Checking that YAML_CONFIG includes only one method from each module...

    Doing a sanity check first...
    Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...
    Sanity check of the YAML_CONFIG was successfully done...

    'One method per module' check was also successfully done...

    Checking that the first method in the pipeline is a loader...
    Loader check successful!!

    'tomopy.misc.corr/median_filters' is not a valid path to a method. Please recheck the yaml file.


This is because :code:`median_filters` is not a valid method in TomoPy -- should be :code:`median_filter`.
To make sure you pass the correct method, refer to the documentation of the package you are using (TomoPy, HTTomoLib, etc.)


What else do we check with the YAML checker?
============================================

* We do a sanity check first, to make sure that the YAML_CONFIG is properly indented and has valid mappings.

For instance, we cannot have the following in a YAML file:

.. code-block:: yaml

    - httomo.data.hdf.loaders:
        standard_tomo:
            name: tomo
          data_path: /entry1/tomo_entry/data/data
          image_key_path: /entry1/tomo_entry/instrument/detector/image_key
          dimension: 1
          pad: 0

This will raise a warning because :code:`name` is not in the same indentation level as :code:`data_path` and :code:`image_key_path`.

* We also check that the tags are valid.

For example, we have the :code:`!Sweep` functionality in HTTomo. If you make a typo,
it will raise an error.

* We check that the first method in the pipeline is always a loader from :code:`'httomo.data.hdf.loaders'`.
* The following will raise an error because if two methods that are adjacent in a pipeline happen to be from the same package,
you cannot omit the package name from the second method's config and you must still provide it.

.. code-block:: yaml

    - tomopy.prep.normalize:
        normalize:
          data_in: tomo
          data_out: tomo
          cutoff: null
        minus_log:
          data_in: tomo
          data_out: tomo

* We check that the parameters for each method are valid. For example, :code:`find_center_vo` method from :code:`tomopy.recon.rotation`
takes :code:`ratio` as a parameter with a float value. If you pass a string instead, it will raise an error. Again the trick is to
refer the documentation always.

* If you pass :code:`IN_DATA` (path to the data) along with the yaml config, as:

.. code-block:: console

    $ python -m httomo check config.yaml IN_DATA

That will check that the paths to the data and keys in the :code:`YAML_CONFIG` file match the paths and keys in the input file (:code:`IN_DATA`).

If you have the following loader in your yaml file:
.. code-block:: yaml

    - httomo.data.hdf.loaders:
        standard_tomo:
          name: tomo
          data_path: entry1/tomo_entry/data/
          image_key_path: entry1/tomo_entry/instrument/detector/image_key
          preview:
            -
            - start: 30
              stop: 60
            -
          pad: 0

And you pass it with standard tomo data, it will raise an error because the paths do not match:

.. code-block:: console

    Checking that YAML_CONFIG includes only one method from each module...

    Doing a sanity check first...
    Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...
    Sanity check of the YAML_CONFIG was successfully done...

    'One method per module' check was also successfully done...

    Checking that the first method in the pipeline is a loader...
    Loader check successful!!

    Checking that the paths to the data and keys in the YAML_CONFIG file match the paths and keys in the input file (IN_DATA)...
    'entry1/tomo_entry/data' is not a valid path to a dataset in YAML_CONFIG. Please recheck the yaml file.


We have many other checks and are constantly improving the YAML checker to make it more robust, verbose, and user-friendly.
This is a user-interface so suggestions are always welcome.
