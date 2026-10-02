import sys
import os
import shutil
from pathlib import Path


def create_md_from_changelog(input_file, output_file):

    with open(input_file, 'r') as file:
        lines = file.readlines()

    md_content = ["# Changelog\n\n"]

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("ASPECT") and "ASPECT" in stripped:
            version_type, version_number, version_date = stripped.split('-')
            md_content.append(f"## {version_number.strip()} {version_type.strip()} ({version_date.strip()})\n")

        elif stripped.startswith("-"):
            md_content.append(f"- {stripped[1:].strip()}\n")

        elif stripped:
            md_content.append(f"**{stripped}**\n")

        else:
            md_content.append("\n")

    with open(output_file, 'w') as file:
        file.writelines(md_content)

    return


# -- Project information -----------------------------------------------------

project = 'aspect'
copyright = '2026, Vital-Fernandez'
author = 'Vital-Fernandez'

release = "1.0.dev0"

# -- General configuration ---------------------------------------------------
_lib_path = Path(__file__).parents[2]/'src'
sys.path.insert(0, os.path.abspath(_lib_path))

extensions = ['myst_nb',
              'sphinx_togglebutton',
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'matplotlib.sphinxext.plot_directive',
              ]

myst_enable_extensions = ["amsmath",
                          "dollarmath",
                          "colon_fence",
                          "html_admonition",
                          "html_image",
                          "deflist",
                          "smartquotes"]

myst_heading_anchors = 2

source_suffix = {'.rst': 'restructuredtext',
                 '.ipynb': 'myst-nb',
                 '.myst': 'myst-nb',
                 '.md': 'myst-nb'}

nbsphinx_markdown = True
nbsphinx_allow_errors = True
nbsphinx_execute = 'always'

autodoc_member_order = 'bysource'
autodoc_default_options = {"imported-members": True}

templates_path = ['_templates', '_build']
exclude_patterns = []

nb_execution_mode = "off"

# Cross-reference the LiMe documentation
intersphinx_mapping = {"lime": ("https://lime-stable.readthedocs.io/en/latest/", None),
                       "python": ("https://docs.python.org/3", None),
                       "numpy": ("https://numpy.org/doc/stable/", None)}

# -- MathJax configuration ---------------------------------------------------

mathjax3_config = {"tex": {"macros": {"textsc": ["{\\small \\uppercase{#1}}", 1]}}}

imgmath_latex_preamble = r'\usepackage[active]{preview}'
imgmath_use_preview = True

# -- HTML output configuration -----------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {"logo": {"image_light": "0_resources/images/aspect_log_bright_no_bg.png",
                               "image_dark":  "0_resources/images/aspect_log_bright_no_bg.png",
                               "alt_text":    "ASPECT Documentation"},
                      "secondary_sidebar_items": [],
                      }

# -- Folder cleanup and example copy -----------------------------------------

_doc_folder = Path(__file__).parents[2]/'docs/source'
_examples_path = Path(__file__).parents[2]/'examples/doc_notebooks'
_examples_path.mkdir(parents=True, exist_ok=True)

sys.path.append(_examples_path.as_posix())

# -- Changelog compiler ------------------------------------------------------
input_txt_changelog = _lib_path/'aspect/changelog.txt'
output_md_changelog = _doc_folder/'4_references/1_changelog.md'
if input_txt_changelog.is_file():
    create_md_from_changelog(input_txt_changelog, output_md_changelog)