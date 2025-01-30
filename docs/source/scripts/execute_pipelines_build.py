#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team <scientificsoftware@diamond.ac.uk>
# Created Date: 30/January/2025
# version ='0.1'
# ---------------------------------------------------------------------------
"""Executing full-pipeline generation for HTTomo using YAML templates from httomo-backends
and yaml_pipelines_generator script available also in httomo-backends.
The built pipelines are placed in HTTomo documentation (docs/source/pipelines_full/) to be
used by tests and documentation build.
"""

import os
import glob
import httomo_backends
import httomo
from httomo_backends.scripts.yaml_pipelines_generator import yaml_pipelines_generator

path_to_httomobackends = os.path.dirname(httomo_backends.__file__)
path_to_httomo = os.path.dirname(httomo.__file__)

path_to_httomo_pipelines = path_to_httomo + "docs/source/pipelines_full/"
pipelines_folder = path_to_httomobackends + "/pipelines_full/"

# loop over all pipeline directive files and running the generator
for filepath in glob.iglob(pipelines_folder + "*.yaml"):
    basename = os.path.basename(filepath)
    outputfile_name = os.path.normpath(basename.replace(r"_directive", r""))
    yaml_pipelines_generator(
        filepath, path_to_httomobackends, path_to_httomo_pipelines + outputfile_name
    )
