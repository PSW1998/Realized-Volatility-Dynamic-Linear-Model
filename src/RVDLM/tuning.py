# src/RVDLM/tuning.py

import numpy as np
import pandas as pd
from .rvdlm_univariate import RVDLM_Univariate

def tune_rvdlm_ohlc(
    df: pd.DataFrame,
    m0: np.ndarray,
    C0: np.ndarray,
    n0: float,
    s0: float,
    beta_grid=None,
    alpha_grid=None,
    lambda_grid=None,
    use_leverage=False,
):
    if beta_grid is None:
        beta_grid = np.linspace(0.7, 0.8, 5)
    if alpha_grid is None:
        alpha_grid = np.linspace(1.0, 5.0, 5)
    if lambda_grid is None:
        lambda_grid = np.linspace(0.95, 0.99, 5)

    best_nll = np.inf
    best_params = None

    for beta in beta_grid:
        for alpha in alpha_grid:
            for lam in lambda_grid:
                model = RVDLM_Univariate(
                    beta=beta,
                    alpha=alpha,
                    lambda_theta=lam,
                    m0=m0,
                    C0=C0,
                    n0=n0,
                    s0=s0,
                    use_leverage=use_leverage,
                )
                nll = model.neg_loglik(df)
                if nll < best_nll:
                    best_nll = nll
                    best_params = (beta, alpha, lam)

    return best_params, best_nll
