# src/RVDLM/rvdlm_univariate.py

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .dynamic_gamma import DynamicGammaFilter

class RVDLM_Univariate:
    """
    Univariate RV–DLM: price + realized-precision variance block.

    Price block: y_t = [1, y_{t-1}, x_t] θ_t + ε_t, ε_t ~ N(0, φ_t^{-1})
    Variance block: φ_t via DynamicGammaFilter; z_t = 1/RV_t.

    This is the main model used in Section 'Empirical Application'.
    """

    def __init__(
        self,
        beta: float,
        alpha: float,
        lambda_theta: float,
        m0: np.ndarray,
        C0: np.ndarray,
        n0: float,
        s0: float,
        use_leverage: bool = False,
    ):
        self.beta = beta
        self.alpha = alpha
        self.lambda_theta = lambda_theta
        self.m0 = m0
        self.C0 = C0
        self.gamma_filter = DynamicGammaFilter(n0, s0, beta, alpha)
        self.use_leverage = use_leverage

    def _design_vector(self, y_prev: float, z_t: float):
        """
        Build F_t. If use_leverage=True, include x_t = z_t^{-1/2}.
        """
        if self.use_leverage:
            x_t = 1.0 / np.sqrt(z_t)
            return np.array([1.0, y_prev, x_t])
        else:
            return np.array([1.0, y_prev])

    def loglik(self, df: pd.DataFrame) -> float:
        """
        One-step-ahead log likelihood of y_t given (y_{t-1}, z_t).

        df must have columns 'Close' and 'precision' (z_t = 1/RV_t).
        """
        y = df["Close"].values
        z = df["precision"].values

        m_post = self.m0.copy()
        C_post = self.C0.copy()
        gf = self.gamma_filter

        loglike = 0.0

        for t in range(1, len(y)):
            # state prior
            m_prior = m_post
            C_prior = C_post / self.lambda_theta

            # variance block
            df_t = gf.n
            sigma2 = gf.s

            F_t = self._design_vector(y[t - 1], z[t])
            yhat = F_t @ m_prior
            Q = F_t @ C_prior @ F_t + sigma2

            loglike += student_t.logpdf(
                y[t],
                df=df_t,
                loc=yhat,
                scale=np.sqrt(Q),
            )

            # state update
            A = (C_prior @ F_t) / Q
            m_post = m_prior + A * (y[t] - yhat)
            C_post = C_prior - np.outer(A, A) * Q
            C_post = (C_post + C_post.T) / 2

            # variance update
            gf.update(z[t])

        return loglike

    def neg_loglik(self, df: pd.DataFrame) -> float:
        return -self.loglik(df)
