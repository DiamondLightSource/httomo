.. _utilities_yamlchecker:

YAML Checker - Why use it?
**************************
YAML checker will help you to validate your process list (see :ref:`explanation_process_list`)
saved as a YAML file. Before running your pipeline with HTTomo, we highly recommend that you validate your process list using this utility. **The checker will help you to identify errors in your process list and avoid problems during the run**.

Usage
=====

.. code-block:: console

   $ python -m httomo check YAML_CONFIG IN_DATA


.. note::

    - Use this :code:`check` command before you use the :code:`run` command to run your pipeline.
    - The :code:`YAML_CONFIG` is the path to your YAML file and :code:`IN_DATA` is the path to your input data.
    - :code:`IN_DATA` is optional, but if you provide it, the yaml checker will be checking that the paths
      to the data and keys in the :code:`YAML_CONFIG` file match the paths and keys in the input file (:code:`IN_DATA`).


For example, if you have the following as a :code:`YAML_CONFIG` file saved as :code:`example.yaml`:

.. literalinclude:: ../../../tests/samples/pipeline_template_examples/testing/example.yaml
   :language: yaml

And you run the YAML checker with:

.. code-block:: console

   $ python -m httomo check example.yaml


You will get the following output:

.. code-block:: console

    Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...
    Sanity check of the YAML_CONFIG was successfully done...

    Checking that the first method in the pipeline is a loader...
    Loader check successful!!


    YAML validation successful!! Please feel free to use the `run` command to run the pipeline.


The Yaml check was successful here because your yaml file was properly indented and had valid mappings and tags.
It also included valid parameters for each method used from TomoPy, HTTomolib, or other `backends <https://diamondlightsource.github.io/httomo/backends/list.html>`_.



But if you had the following as a :code:`YAML_CONFIG` file saved as :code:`incorrect_method.yaml`:

.. literalinclude:: ../../../tests/samples/pipeline_template_examples/testing/incorrect_method.yaml
   :language: yaml

And then you run the YAML checker, you get:

.. code-block:: console

    $ python -m httomo check incorrect_method.yaml
    Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...
    Sanity check of the YAML_CONFIG was successfully done...

    Checking that the first method in the pipeline is a loader...
    Loader check successful!!

    'tomopy.misc.corr/median_filters' is not a valid method. Please recheck the yaml file.


This is because :code:`median_filters` is not a valid method in TomoPy -- should be :code:`median_filter`.
To make sure you pass the correct method, refer to the documentation of the package you are using (TomoPy, HTTomoLib, etc.)


What else do we check with the YAML checker?
============================================

* We do a sanity check first, to make sure that the YAML_CONFIG is properly indented and has valid mappings.

For instance, we cannot have the following in a YAML file:

.. literalinclude:: ../../../tests/samples/pipeline_template_examples/testing/wrong_indentation_pipeline.yaml
   :language: yaml

This will raise a warning because :code:`name` is not in the same indentation level as :code:`data_path` and :code:`image_key_path`.

* We check that the first method in the pipeline is always a loader from :code:`'httomo.data.hdf.loaders'`.
* We check methods exist for the given module path.
* We check that the parameters for each method are valid. For example, :code:`find_center_vo` method from :code:`tomopy.recon.rotation` takes :code:`ratio` as a parameter with a float value. If you pass a string instead, it will raise an error. Again the trick is to refer the documentation always.
* We check the required parameters for each method are present.
* If you pass :code:`IN_DATA` (path to the data) along with the yaml config, as:

.. code-block:: console

    $ python -m httomo check config.yaml IN_DATA

That will check that the paths to the data and keys in the :code:`YAML_CONFIG` file match the paths and keys in the input file (:code:`IN_DATA`).

If you have the following loader in your yaml file:

.. literalinclude:: ../../../tests/samples/pipeline_template_examples/testing/incorrect_path.yaml
   :language: yaml

And you provide that, together with the standard tomo data, it will raise an error because the image path does not match:

.. code-block:: console

    Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...
    Sanity check of the YAML_CONFIG was successfully done...

    Checking that the first method in the pipeline is a loader...
    Loader check successful!!

    Checking that the paths to the data and keys in the YAML_CONFIG file match the paths and keys in the input file (IN_DATA)...
    'entry1/tomo_entry/instrument/detect/image_key' is not a valid path to a dataset in YAML_CONFIG. Please recheck the yaml file.

We have many other checks and are constantly improving the YAML checker to make it more robust, verbose, and user-friendly.
This is a user-interface so suggestions are always welcome.
