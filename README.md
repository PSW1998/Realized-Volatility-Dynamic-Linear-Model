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

Clone the repo and install in editable mode:

```bash
git clone https://github.com/PSW1998/Realized-Volatility-Dynamic-Linear-Model.git
cd Realized-Volatility-Dynamic-Linear-Model
pip install -e .
