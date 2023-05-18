# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#!/usr/bin/env python
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
    "numpy",
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

template_patterns = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
#html_theme_path = [sphinx-rtd-theme.get_html_theme_path()]
html_logo = "images/logo/httomo_logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
    "githuburl": "https://github.com/DiamondLightSource/httomo",
}

html_use_smartypants = True
