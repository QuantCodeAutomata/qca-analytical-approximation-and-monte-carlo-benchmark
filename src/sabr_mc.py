"""
Lognormal SABR Monte Carlo simulation.

Dynamics:
  dS_t = sigma_t * S_t * dW_t
  d sigma_t = nu * sigma_t * dB_t
  d<W,B>_t = rho dt

Discretisation (Euler):
  sigma_{i+1} = sigma_i * exp(nu * DeltaB_i - 0.5*nu^2*Delta_t)
  S_{i+1} = S_i * (1 + sigma_i * DeltaW_i)

Correlated increments: DeltaW = sqrt(dt)*Z1, DeltaB = sqrt(dt)*(rho*Z1 + sqrt(1-rho^2)*Z2)
where Z1, Z2 ~ N(0,1) independent.

Only terminal values and running maxima are tracked (streaming, no full path storage).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def simulate_sabr_batch(
    n_paths: int,
    n_steps: int,
    nu: float,
    rho: float,
    sigma0: float,
    S0: float,
    B: float,
    theta: float,
    seed: int | None = None,
    batch_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate SABR paths returning (S_T, S_max_full, S_max_half).

    S_max_full: running max over all 25000 endpoints.
    S_max_half: running max over every 2nd endpoint (12500 dates).
    """
    rng = np.random.default_rng(seed)
    dt = theta / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(1.0 - rho**2)
    nu_sq_half = 0.5 * nu**2 * dt

    S_T = np.zeros(n_paths)
    S_max_full = np.zeros(n_paths)
    S_max_half = np.zeros(n_paths)

    start = 0
    while start < n_paths:
        end = min(start + batch_size, n_paths)
        nb = end - start

        S = np.full(nb, S0)
        sig = np.full(nb, sigma0)
        S_mx_full = np.full(nb, S0)
        S_mx_half = np.full(nb, S0)

        for step_i in range(n_steps):
            Z1 = rng.standard_normal(nb)
            Z2 = rng.standard_normal(nb)
            dW = sqrt_dt * Z1
            dB = sqrt_dt * (rho * Z1 + sqrt_1_rho2 * Z2)

            # Update stock first using sigma_i (current vol, per paper spec)
            S = S * (1.0 + sig * dW)
            # Then update volatility (exact lognormal step)
            sig = sig * np.exp(nu * dB - nu_sq_half)
            # Track maxima
            S_mx_full = np.maximum(S_mx_full, S)
            if (step_i + 1) % 2 == 0:  # every 2nd step
                S_mx_half = np.maximum(S_mx_half, S)

        S_T[start:end] = S
        S_max_full[start:end] = S_mx_full
        S_max_half[start:end] = S_mx_half
        start = end

    return S_T, S_max_full, S_max_half


def compute_payoffs(
    S_T: np.ndarray,
    S_max: np.ndarray,
    K_grid: np.ndarray,
    B: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute up-and-out put payoffs and control payoff."""
    survived = (S_max < B).astype(float)
    Y_put = np.maximum(K_grid[None, :] - S_T[:, None], 0.0) * survived[:, None]
    Y_ctrl = np.maximum(B - S_T, 0.0) * survived
    return Y_put, Y_ctrl
