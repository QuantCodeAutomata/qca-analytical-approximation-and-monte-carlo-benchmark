"""
Error metrics for comparing analytical approximations against Monte Carlo benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_errors(
    prices_analytical: dict[str, np.ndarray],
    mc_prices: np.ndarray,
    K_grid: np.ndarray,
) -> pd.DataFrame:
    """Compute signed error, absolute error, MAE, and max AE for each method.

    Parameters
    ----------
    prices_analytical : dict
        Mapping method_name -> price array of shape (n_strikes,).
    mc_prices : np.ndarray, shape (n_strikes,)
        Monte Carlo reference prices.
    K_grid : np.ndarray
        Strike grid.

    Returns
    -------
    pd.DataFrame with columns: Strike, Method, Price, MC_Price, SignedError,
        AbsError, MAE, MaxAE.
    """
    rows = []
    for method, prices in prices_analytical.items():
        signed_err = prices - mc_prices
        abs_err = np.abs(signed_err)
        mae = float(np.mean(abs_err))
        max_ae = float(np.max(abs_err))
        for i, K in enumerate(K_grid):
            rows.append(
                {
                    "Strike": K,
                    "Method": method,
                    "Price": prices[i],
                    "MC_Price": mc_prices[i],
                    "SignedError": signed_err[i],
                    "AbsError": abs_err[i],
                    "MAE": mae,
                    "MaxAE": max_ae,
                }
            )
    return pd.DataFrame(rows)


def monitoring_sensitivity(
    price_full: np.ndarray,
    price_half: np.ndarray,
    K_grid: np.ndarray,
    se_full: Optional[np.ndarray] = None,
    se_half: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compare MC prices under full vs half monitoring frequency.

    Parameters
    ----------
    price_full : np.ndarray
        MC prices with all monitoring dates.
    price_half : np.ndarray
        MC prices with every-second endpoint (half frequency).
    K_grid : np.ndarray
        Strike grid.
    se_full : optional, std errors for full monitoring.
    se_half : optional, std errors for half monitoring.

    Returns
    -------
    pd.DataFrame with monitoring sensitivity diagnostics.
    """
    diff = price_full - price_half
    abs_diff = np.abs(diff)

    rows = []
    for i, K in enumerate(K_grid):
        row = {
            "Strike": K,
            "Price_25k": price_full[i],
            "Price_12k": price_half[i],
            "Diff": diff[i],
            "AbsDiff": abs_diff[i],
        }
        if se_full is not None and se_half is not None:
            paired_se = np.sqrt(se_full[i] ** 2 + se_half[i] ** 2)
            row["PairedSE"] = paired_se
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["max_abs_diff"] = float(abs_diff.max())
    if se_full is not None and se_half is not None:
        paired_se_arr = np.sqrt(se_full ** 2 + se_half ** 2)
        df.attrs["max_paired_se"] = float(paired_se_arr.max())
    return df
