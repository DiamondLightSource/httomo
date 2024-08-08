#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2023 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ecpress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>

# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date
from unittest import mock

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("../.."))

# -- Mock imports -------------------------------------------------------------


# Mock imports instead of full environment in readthedocs
MOCK_MODULES = [
    "click",
    "mpi4py",
    "cupy",
    "h5py",
    "yaml",
    "skimage",
    "skimage.exposure",
    "nvtx",
    "mpi4py.MPI",
    "scipy",
    "scipy.ndimage",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# ------------------------------------------------------------------------------

project = "HTTomo"
copyright = f"{date.today().year}, Diamond Light Source"

# Specify a base language to help assistive technology
language = "en"

# Save the commit hash, this is displayed in the page title
release = os.popen('git log -1 --format="%H"').read().strip()

# Set version as the latest tag in the current branch
version = os.popen("git describe --tags --abbrev=0").read().strip()

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    # Generates api files
    "sphinx.ext.autodoc",
    # Generates a short summary from docstring
    "sphinx.ext.autosummary",
    # Allows parsing of google style docstrings
    "sphinx.ext.napoleon",
    # Add links to highlighted source code
    "sphinx.ext.viewcode",
    # Allows a grid layout and dropdown boxes
    "sphinx_panels",
    # copy to clipboard button
    "sphinx_copybutton",
    # use jupyter notebooks
    "nbsphinx",
    #'IPython.sphinxext.ipython_console_highlighting',
    "sphinx.ext.githubpages",
    # Generate .nojekyll file for git pages build
]

autosummary_generate = True
numfig = True
template_patterns = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_book_theme"
html_logo = "_static/logo_light.png"
html_title = "HTTomo Documentation page"
html_copy_source = True
html_favicon = "_static/logo_light.png"
html_last_updated_fmt = ""
html_static_path = ["_static"]
html_use_smartypants = True

html_theme_options = {
    "logo": {
        "image_dark": "_static/logo_dark.png",
    },
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
}

html_context = {
    "github_user": "HTTomo",
    "github_repo": "https://github.com/DiamondLightSource/httomo",
    "github_version": "main",
    "doc_path": "docs",
}


def setup(app):
    app.add_css_file("css/general.css")
