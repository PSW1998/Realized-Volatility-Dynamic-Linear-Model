"""
RVDLM: Realized Volatility Dynamic Linear Models

Companion code for RV–DLM paper.
"""

from importlib.metadata import PackageNotFoundError, version

# Package version (if installed via pip); fallback for editable dev
try:
    __version__ = version("RVDLM")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Expose submodules at the top level
from . import dynamic_gamma
from . import rvdlm_univariate
from . import tuning
# If you added dlm_baseline.py:
try:
    from . import dlm_baseline  # optional, if file exists
except ImportError:  # pragma: no cover
    dlm_baseline = None

__all__ = [
    "__version__",
    "dynamic_gamma",
    "rvdlm_univariate",
    "tuning",
    "dlm_baseline",
]

