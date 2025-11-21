# src/RVDLM/dlm_baseline.py

import numpy as np
from scipy.stats import t as student_t


def dlm_lag1_logpred(
    y: np.ndarray,
    lambda_theta: float,
    lambda_sigma: float,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
    nu0: float = 2.0,
    S0: float = 0.001,
) -> np.ndarray:
    """
    Classical discount DLM with AR(1) mean and stochastic volatility.

    y_t = [1, y_{t-1}]' θ_t + ε_t,  ε_t ~ N(0, σ_t^2)
    σ_t^2 evolves via inverse-gamma discounting.

    Returns
    -------
    logpred : np.ndarray
        One-step-ahead log predictive density log p(y_t | D_{t-1})
        for t = 0,...,n-1 (logpred[0] is usually unused).
    """
    n = len(y)
    logpred = np.zeros(n)

    # State: [intercept, AR(1)]
    m_post = m0.copy() if m0 is not None else np.array([0.0, 0.9])
    C_post = C0.copy() if C0 is not None else np.eye(2) * 0.1
    nu_post, S_post = nu0, S0

    for t in range(1, n):
        # Prior for state and variance
        C_prior = C_post / lambda_theta
        nu_prior = lambda_sigma * nu_post
        S_prior = lambda_sigma * S_post
        sigma2 = S_prior / nu_prior

        # Forecast
        F_t = np.array([1.0, y[t - 1]])
        yhat = F_t @ m_post
        Q = F_t @ C_prior @ F_t + sigma2

        logpred[t] = student_t.logpdf(
            y[t],
            df=nu_prior,
            loc=yhat,
            scale=np.sqrt(Q),
        )

        # Posterior update
        e_t = y[t] - yhat
        nu_post = nu_prior + 1
        S_post = S_prior + (S_prior / nu_prior) * (e_t**2) / Q

        A = (C_prior @ F_t) / Q
        m_post = m_post + A * e_t
        C_post = C_prior - np.outer(A, A) * Q
        C_post = (C_post + C_post.T) / 2

    return logpred


def grid_search_dlm(
    y: np.ndarray,
    m0: np.ndarray,
    C0: np.ndarray,
    nu0: float,
    S0: float,
    lambda_theta_grid=None,
    lambda_sigma_grid=None,
):
    """
    Grid search over (lambda_theta, lambda_sigma)
    for the classical DLM, using one-step log predictive likelihood.

    Returns
    -------
    best_params : (lambda_theta, lambda_sigma)
    best_nll    : float (negative log likelihood)
    """
    if lambda_theta_grid is None:
        lambda_theta_grid = np.linspace(0.95, 0.99, 5)
    if lambda_sigma_grid is None:
        lambda_sigma_grid = np.linspace(0.85, 0.95, 5)

    best_nll = np.inf
    best_p = None

    for lt in lambda_theta_grid:
        for ls in lambda_sigma_grid:
            ll = dlm_lag1_logpred(y, lt, ls, m0=m0, C0=C0, nu0=nu0, S0=S0)
            nll = -np.sum(ll[1:])  # drop t=0
            if nll < best_nll:
                best_nll = nll
                best_p = (lt, ls)

    return best_p, best_nll
