# src/RVDLM/__init__.py

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("RVDLM")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .dynamic_gamma import DynamicGammaFilter  # or GammaF_Filter if that's the name
from .rvdlm_univariate import (
    combined_neg_loglike_univariate_lag1,
    grid_search_univariate_lag1,
    # plus anything else you defined there
)
from .tuning import tune_rvdlm_ohlc  # if you have it
from .dlm_baseline import dlm_lag1_logpred, grid_search_dlm  # <-- NEW

__all__ = [
    "__version__",
    "DynamicGammaFilter",
    "combined_neg_loglike_univariate_lag1",
    "grid_search_univariate_lag1",
    "tune_rvdlm_ohlc",
    "dlm_lag1_logpred",
    "grid_search_dlm",
]

