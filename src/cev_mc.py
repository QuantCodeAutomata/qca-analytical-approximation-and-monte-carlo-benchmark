"""
CEV (Constant Elasticity of Variance) Monte Carlo simulation.

Dynamics: dS_t = sigma * S_t^beta * dW_t  (zero drift, pricing measure)
Discretisation: absorbed Euler – S_{i+1} = max(S_i + sigma*S_i^beta*DeltaW_i, 0)
Once S hits zero it stays zero.

Only terminal value and running maximum are tracked (no full path storage).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def simulate_cev_batch(
    n_paths: int,
    n_steps: int,
    sigma: float,
    beta: float,
    S0: float,
    B: float,
    theta: float,
    seed: int | None = None,
    batch_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate CEV paths and return (S_terminal, S_running_max).

    Parameters
    ----------
    n_paths : int
    n_steps : int  (e.g. 25000)
    sigma : float  CEV diffusion coefficient
    beta : float   CEV exponent
    S0 : float     Initial spot
    B : float      Barrier
    theta : float  Maturity
    seed : int | None
    batch_size : int  Number of paths per memory batch

    Returns
    -------
    S_T : ndarray (n_paths,)  terminal stock price
    S_max : ndarray (n_paths,)  running maximum
    """
    rng = np.random.default_rng(seed)
    dt = theta / n_steps
    sqrt_dt = np.sqrt(dt)

    S_T = np.zeros(n_paths)
    S_max = np.zeros(n_paths)

    start = 0
    while start < n_paths:
        end = min(start + batch_size, n_paths)
        nb = end - start

        S = np.full(nb, S0)
        S_mx = np.full(nb, S0)

        for _ in range(n_steps):
            dW = rng.standard_normal(nb) * sqrt_dt
            S = np.maximum(S + sigma * (S ** beta) * dW, 0.0)
            S_mx = np.maximum(S_mx, S)

        S_T[start:end] = S
        S_max[start:end] = S_mx
        start = end

    return S_T, S_max


def simulate_cev_batch_dual(
    n_paths: int,
    n_steps: int,
    sigma: float,
    beta: float,
    S0: float,
    B: float,
    theta: float,
    seed: int | None = None,
    batch_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate CEV paths tracking two running maxima simultaneously.

    S_max_full: running max checked at every step (n_steps monitoring dates).
    S_max_half: running max checked at every 2nd step (n_steps//2 monitoring dates).

    Parameters
    ----------
    n_paths : int
    n_steps : int  Must be even (e.g. 25000).
    sigma : float  CEV diffusion coefficient
    beta : float   CEV exponent
    S0 : float     Initial spot
    B : float      Barrier
    theta : float  Maturity
    seed : int | None
    batch_size : int  Number of paths per memory batch

    Returns
    -------
    S_T       : ndarray (n_paths,)  terminal stock price
    S_max_full: ndarray (n_paths,)  running max over all steps
    S_max_half: ndarray (n_paths,)  running max over every 2nd step
    """
    rng = np.random.default_rng(seed)
    dt = theta / n_steps
    sqrt_dt = np.sqrt(dt)

    S_T = np.zeros(n_paths)
    S_max_full = np.zeros(n_paths)
    S_max_half = np.zeros(n_paths)

    start = 0
    while start < n_paths:
        end = min(start + batch_size, n_paths)
        nb = end - start

        S = np.full(nb, S0)
        S_mx_full = np.full(nb, S0)
        S_mx_half = np.full(nb, S0)

        for step_i in range(n_steps):
            dW = rng.standard_normal(nb) * sqrt_dt
            S = np.maximum(S + sigma * (S ** beta) * dW, 0.0)
            S_mx_full = np.maximum(S_mx_full, S)
            if (step_i + 1) % 2 == 0:  # every 2nd step: steps 2, 4, ..., n_steps
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
    """Compute up-and-out put payoffs and control payoff.

    Parameters
    ----------
    S_T : ndarray (n_paths,)  terminal values
    S_max : ndarray (n_paths,)  running maxima (full or half monitoring)
    K_grid : ndarray (n_strikes,)
    B : float  barrier

    Returns
    -------
    Y_put : ndarray (n_paths, n_strikes)  up-and-out put payoffs
    Y_ctrl : ndarray (n_paths,)  control payoff (B - S_T)^+ * survived
    """
    survived = (S_max < B).astype(float)
    Y_put = np.maximum(K_grid[None, :] - S_T[:, None], 0.0) * survived[:, None]
    Y_ctrl = np.maximum(B - S_T, 0.0) * survived
    return Y_put, Y_ctrl
