# Realized-Volatility-Dynamic-Linear-Model

Reference implementation for

> **“Bayesian dynamic modelling of realized volatility in financial asset return forecasting”**  
> Patrick Woitschig & Mike West

This repo provides a small, focused Python package for **Bayesian dynamic models of realized volatility**. It implements a conjugate **dynamic gamma / scaled-F variance block** and embeds it into a **dynamic linear model (DLM)** for asset prices, so that:

- filtering and forecasting are **fully analytic** (no MCMC / particles),
- realized variance (or precision) is used as a **noisy measurement of latent volatility**, and  
- one-step predictive distributions are **Student-t for prices** and **scaled-F for realized variance**.

The core applied setting is S&P 500 sector ETFs with realized measures (e.g. Rogers–Satchell RV, intraday RV/RQ).

---

## Installation

You don’t need to clone the repo if you just want to use the package:

```bash
pip install "git+https://github.com/PSW1998/Realized-Volatility-Dynamic-Linear-Model.git"
```

## Examples

An end-to-end example is provided as a Jupyter notebook:

- [`examples/SPYExample.ipynb`](examples/SPY_Example.ipynb)  
  Downloads daily SPY data from Yahoo Finance, constructs a Rogers–Satchell realized variance and precision series, tunes an RV–DLM via `tune_rvdlm_ohlc`, tunes a classical discount DLM via `grid_search_dlm`, and compares one-step-ahead predictive log-likelihoods.
