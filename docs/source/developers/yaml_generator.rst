.. _developers_yamlgenerator:

YAML generator
**************************
The templates for :ref:`backends_content` are generated automatically using the YAML generator tool `provided <https://github.com/DiamondLightSource/httomo/blob/main/templates/yaml_templates_generator.py>`_ in the Github repo. 

The code does the following: 

 * Generates a list of YAML files for all accessible methods in a chosen software package, such as TomoPy.
 * Modifies some extracted parameters in YAML templates to work with HTTomo smoothly.
 
How does it work:

* You would need to provide a YAML file (e.g. *modules.yaml*) with the listed modules you would like to inspect and extract methods from, for instance for TomoPy this could be:

.. code-block:: yaml

    - tomopy.misc.corr
    - tomopy.misc.morph

* Then you run the generator using the following command:
 
.. code-block:: console
   
   python -m yaml_templates_generator -m /path/to/modules.yaml -o /path/to/outputfolder/

* Please note that the package (e.g. TomoPy) must be installed into your conda environment and therefore be accessible when importing.