# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# from elastica.version import sopht_version

from sopht.version import sopht_version

project = "SophT"
copyright = "2023, Gazzola Lab"
author = (
    "Yashraj Bhosale, "
    "Arman Tekinalp, "
    "Songyuan Cui, "
    "Fan Kiat Chan, "
    "Mattia Gazzola"
)

release = sopht_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
]

templates_path = ["_templates"]
# exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
