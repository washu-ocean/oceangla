# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../../src/"))

project = 'oceangla'
copyright = '2026, Ramone Agard, Joey Scanga'
author = 'Ramone Agard, Joey Scanga'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = [
    "sphinx.ext.autodoc",   # Core library for docstring extraction
    "sphinx.ext.napoleon",  # Allows Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Adds links to highlighted source code
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
