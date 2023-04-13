## YAML template generator:
 * Generates a list of YAML files for all accessible methods in a software package, such as TomoPy.
 * Modifies some extracted parameters to work with httomo smoothly.
 
### How does it work:
* You would need to provide a YAML file (`modules.yaml`) with the listed modules which you would like to inspect and extract methods from, for instance:
```
- tomopy.misc.corr
- tomopy.misc.morph
```
* Then you run the generator with 
```
python -m yaml_templates_generator -m /path/to/modules.yaml -o /path/to/output/
```
* Please note that the package (e.g. TomoPy) must be installed into your conda environment and be accessible.