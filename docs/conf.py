"""Sphinx configuration for Chalcedon documentation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "Chalcedon"
author = "Chalcedon contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
autodoc_mock_imports = ["tdpy"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"
language = "en"
html_theme = "sphinx_rtd_theme"