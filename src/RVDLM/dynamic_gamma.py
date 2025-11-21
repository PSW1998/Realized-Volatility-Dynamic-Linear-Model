# src/RVDLM/dynamic_gamma.py

import numpy as np
from scipy.stats import f

class DynamicGammaFilter:
    """
    Dynamic gamma / dynamic-F variance block.

    Implements the Beta–Gamma discount evolution for precision φ_t
    and the scaled-F predictive for realized precision z_t = 1/RV_t.

    This is essentially Section 'Dynamic Gamma Model' in the paper.
    """

    def __init__(self, n0: float, s0: float, beta: float, alpha: float):
        """
        Parameters
        ----------
        n0, s0 : float
            Initial gamma hyperparameters: φ_0 ~ G(n0, n0 s0).
        beta : float
            Discount factor (0 < beta <= 1).
        alpha : float
            Observation shape parameter (>1).
        """
        self.n = n0
        self.s = s0
        self.beta = beta
        self.alpha = alpha

    def prior(self):
        """Return (n_prior, s_prior) at time t given (n_{t-1}, s_{t-1})."""
        n_prior = self.beta * self.n
        s_prior = self.s  # mean 1/s unchanged
        return n_prior, s_prior

    def predictive_params(self):
        """
        Parameters of z_t | D_{t-1}.

        Returns (scale, df1, df2) for:
        z_t = scale * F(df1, df2)
        """
        n_prior, s_prior = self.prior()
        df1 = 2 * n_prior
        df2 = 2 * self.alpha + 2
        scale = self.alpha / ((1 + self.alpha) * s_prior)
        return scale, df1, df2

    def predictive_moments(self):
        """Return mean, median, 95% CI for z_t | D_{t-1}."""
        scale, df1, df2 = self.predictive_params()
        mean = scale * (df2 / (df2 - 2)) if df2 > 2 else np.nan
        median = scale * f.median(df1, df2)
        ci95 = (
            scale * f.ppf(0.025, df1, df2),
            scale * f.ppf(0.975, df1, df2),
        )
        return mean, median, ci95

    def update(self, z_t: float):
        """
        Update (n_t, s_t) with realized precision z_t.

        This is the beta–gamma recursion from the paper:

        n_t = β n_{t-1} + 1 + α
        s_t = (β n_{t-1} s_{t-1} + α / z_t) / n_t
        """
        n_prior = self.beta * self.n
        n_post = n_prior + 1 + self.alpha
        s_post = (n_prior * self.s + self.alpha / z_t) / n_post

        self.n, self.s = n_post, s_post

    # You can later add .smooth() methods that implement the backward recursion
