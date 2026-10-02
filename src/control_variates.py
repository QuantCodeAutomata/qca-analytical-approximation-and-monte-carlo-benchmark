"""
Control-variate Monte Carlo estimation for up-and-out put options.

The control payoff is the (B - S_N)^+ * 1_{max S_i < B} (a digital-barrier
put struck at the barrier B), whose theoretical price is B - S0 when the
barrier is continuously monitored and the stock starts at S0 < B.
We use the endpoint-monitored price B - S0 as the known control expectation.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def fit_control_coefficients(
    Y_put: np.ndarray,
    Y_ctrl: np.ndarray,
) -> np.ndarray:
    """Estimate per-strike control-variate coefficients from pilot paths.

    beta_hat_K = Cov(Y_K, Y_B) / Var(Y_B)

    Parameters
    ----------
    Y_put : np.ndarray, shape (n_pilot, n_strikes)
        Discounted payoffs of the up-and-out put for each strike.
    Y_ctrl : np.ndarray, shape (n_pilot,)
        Control payoff (B - S_N)^+ * 1_{barrier survived}.

    Returns
    -------
    beta_hat : np.ndarray, shape (n_strikes,)
        Estimated control coefficients, one per strike.
    """
    assert Y_put.ndim == 2, "Y_put must be (n_pilot, n_strikes)"
    assert Y_ctrl.ndim == 1, "Y_ctrl must be (n_pilot,)"
    assert Y_put.shape[0] == Y_ctrl.shape[0]

    var_ctrl = np.var(Y_ctrl, ddof=1)
    if var_ctrl < 1e-15:
        return np.zeros(Y_put.shape[1])

    cov = np.mean(
        (Y_put - Y_put.mean(axis=0)[None, :])
        * (Y_ctrl - Y_ctrl.mean())[:, None],
        axis=0,
    )
    return cov / var_ctrl


def controlled_price(
    Y_put: np.ndarray,
    Y_ctrl: np.ndarray,
    beta_hat: np.ndarray,
    S0: float,
    B: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute controlled MC price estimates with standard errors and 95% CIs.

    p_hat_K = mean(Y_K) - beta_hat_K * (mean(Y_B) - (B - S0))

    Parameters
    ----------
    Y_put : np.ndarray, shape (n_paths, n_strikes)
        Put payoffs on production paths.
    Y_ctrl : np.ndarray, shape (n_paths,)
        Control payoffs on production paths.
    beta_hat : np.ndarray, shape (n_strikes,)
        Frozen pilot control coefficients.
    S0 : float
        Spot price.
    B : float
        Barrier level.

    Returns
    -------
    price : np.ndarray, shape (n_strikes,)
    se : np.ndarray, shape (n_strikes,)
    ci_lo : np.ndarray, shape (n_strikes,)
    ci_hi : np.ndarray, shape (n_strikes,)
    """
    n = Y_put.shape[0]
    ctrl_mean = Y_ctrl.mean()
    ctrl_known = B - S0  # Known expectation (endpoint-monitored)

    # Controlled variates for each path
    Y_cv = Y_put - beta_hat[None, :] * (Y_ctrl[:, None] - ctrl_mean)
    # Adjust bias correction: subtract the mean control bias
    price = Y_put.mean(axis=0) - beta_hat * (ctrl_mean - ctrl_known)

    # Standard error of the controlled estimator
    Y_cv_centred = Y_cv - price[None, :]
    se = np.std(Y_cv_centred, ddof=1, axis=0) / np.sqrt(n)

    ci_lo = price - 1.96 * se
    ci_hi = price + 1.96 * se

    return price, se, ci_lo, ci_hi
