#!/bin/bash

python -m pip install .

# Define var for path to httomo installation dir
#
# `SP_DIR` is the path to the `site-packages/` dir in the conda env, see
# https://docs.conda.io/projects/conda-build/en/latest/user-guide/environment-variables.html#environment-variables-set-during-the-build-process
HTTOMO_TARGET="$SP_DIR/$PKG_NAME"

# Create dirs for YAML templates
mkdir $HTTOMO_TARGET/yaml_templates/
mkdir $HTTOMO_TARGET/yaml_templates/httomo
mkdir $HTTOMO_TARGET/yaml_templates/httomolib
mkdir $HTTOMO_TARGET/yaml_templates/httomolibgpu
mkdir $HTTOMO_TARGET/yaml_templates/tomopy

# Generate YAML templates
python scripts/yaml_templates_generator.py -i httomo/methods_database/packages/httomo_modules.yaml -o $HTTOMO_TARGET/yaml_templates/httomo/
python scripts/yaml_templates_generator.py -i httomo/methods_database/packages/external/httomolib/httomolib_modules.yaml -o $HTTOMO_TARGET/yaml_templates/httomolib/
python scripts/yaml_templates_generator.py -i httomo/methods_database/packages/external/httomolibgpu/httomolibgpu_modules.yaml -o $HTTOMO_TARGET/yaml_templates/httomolibgpu/
python scripts/yaml_templates_generator.py -i httomo/methods_database/packages/external/tomopy/tomopy_modules.yaml -o $HTTOMO_TARGET/yaml_templates/tomopy/
python scripts/yaml_unsupported_tomopy_remove.py -t $HTTOMO_TARGET/yaml_templates/tomopy/ -l httomo/methods_database/packages/external/tomopy/tomopy.yaml
