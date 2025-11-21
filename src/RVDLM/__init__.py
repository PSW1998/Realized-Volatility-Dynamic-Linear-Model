"""
RVDLM: Realized Volatility Dynamic Linear Model

Core tools for fitting and forecasting realized volatility with
dynamic linear models.
"""

from importlib.metadata import PackageNotFoundError, version

# Expose a __version__ attribute that matches the installed package version
try:
    __version__ = version("RVDLM")  # <- must match [project].name in pyproject.toml
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without `pip install -e .`)
    __version__ = "0.0.0"

# If you want a very minimal public API for now, keep __all__ tiny.
__all__ = ["__version__"]
