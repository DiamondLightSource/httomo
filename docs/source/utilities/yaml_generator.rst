.. _utilities_yamlgenerator:

Templates generator
*******************
:ref:`backends_content` can be generated automatically using the YAML generator tool `provided <https://github.com/DiamondLightSource/httomo/tree/main/scripts/yaml_templates_generator.py>`_ in the Github repo.

The script does the following:

* Generates a list of YAML files for all accessible on import methods in a chosen software package, e.g., TomoPy.
* Modifies and/or removes some extracted parameters in YAML templates to make the templates compatible with HTTomo.

How does it work:

* The user would need to provide a YAML file with the listed *modules* you would like to inspect and extract the methods from. For instance, for the TomoPy package this would be:

.. code-block:: yaml

    - tomopy.misc.corr
    - tomopy.misc.morph

* The generator can be applied using the following command:

.. code-block:: console

   $ python -m yaml_templates_generator -i /path/to/modules.yaml -o /path/to/outputfolder/

Please note that the package from which the modules are extracted, must be installed into your conda environment.

**For TomoPy templates only.** After templates have been generated for TomoPy, we need to remove the ones that are not currently supported by HTTomo. We do that by looking into the library file that exists in HTTomo for TomoPy.

.. code-block:: console

   $ python -m remove_unsupported_templates -t /path/to/templates/ -l /path/to/library/file
