# src/your_package_name/__init__.py
"""
your_package_name

Tools for doing XYZ.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("RVDLM")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .core import fit_model, forecast
from .utils import load_data

__all__ = [
    "fit_model",
    "forecast",
    "load_data",
    "__version__",
]
