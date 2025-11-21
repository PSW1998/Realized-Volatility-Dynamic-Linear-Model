# src/RVDLM/__init__.py

"""
RVDLM: Realized Volatility Dynamic Linear Models

Reference implementation for:

  "Bayesian dynamic modelling of realized volatility in financial asset return forecasting"
  by Patrick Woitschig and Mike West.

Provides:
- DynamicGammaFilter: dynamic-F variance block (Sec. Dynamic Gamma Model)
- RVDLM_Univariate: price + realized-variance RV–DLM (Secs. 3–4)
- Quarticity-based α_t construction (Sec. Quarticity–matched effective DOF)
- Baseline DLM and scoring utilities for empirical comparisons.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("RVDLM")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .dynamic_gamma import DynamicGammaFilter
from .rvdlm_univariate import RVDLM_Univariate
from .quarticity import alpha_from_quarticity, alpha_series_from_quarticity
from .tuning import tune_rvdlm_ohlc
from .dlm_baseline import dlm_lag1_logpred, grid_search_dlm
from .scoring import log_score_series, bayes_factor_summary  # to be written

__all__ = [
    "__version__",
    "DynamicGammaFilter",
    "RVDLM_Univariate",
    "tune_rvdlm_ohlc",
    "dlm_lag1_logpred",
    "grid_search_dlm",
]

