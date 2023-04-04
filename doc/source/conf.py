
# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#!/usr/bin/env python
import sys
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath('../../src/httomo'))
sys.path.insert(0, os.path.abspath('../../src'))

# -- Mock imports -------------------------------------------------------------

from unittest import mock

# Mock imports instead of full environment in readthedocs
MOCK_MODULES = ["numpy",
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
                "scipy.ndimage"
                ]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# ------------------------------------------------------------------------------

project = 'HTTomo'
copyright = '2022, Diamond Light Source'

release = os.popen('git log -1 --format="%H"').read().strip()

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
    'sphinx_copybutton',
    # use jupyter notebooks
    'nbsphinx',
    #'IPython.sphinxext.ipython_console_highlighting'
]

autosummary_generate = True

template_patterns = ['_templates']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
