"""
Rough Bergomi Monte Carlo simulation.

Dynamics:
  dS_t / S_t = sqrt(V_t) dB_t^S
  B_t^S = rho W_t + sqrt(1-rho^2) W_t^perp
  V_t = xi0 * exp(nu*sqrt(2H) * int_0^t (t-s)^(H-1/2) dW_s - 0.5*nu^2*t^(2H))

Discretisation (3R hybrid / FFT convolution):
  Z_{t_i} = int_0^{t_i} (t_i - s)^(H-1/2) dW_s
           ~ sum_{j=0}^{i-1} kappa_{i-j} * DeltaW_j
  where kappa_m = (m * dt)^(H-1/2) for m >= 1, kappa_0 = 0.

The discrete Volterra convolution is computed via FFT for efficiency.
Log-price update:
  X_{i+1} = X_i - 0.5*V_i*dt + sqrt(V_i)*(rho*DeltaW_i + sqrt(1-rho^2)*DeltaW_i^perp)
  S_i = S0 * exp(X_i)

Paths are simulated in batches to control memory usage.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def _build_volterra_kernel(n_steps: int, dt: float, H: float) -> np.ndarray:
    """Build discretised Volterra kernel kappa.

    kappa[m] = (m * dt)^(H - 0.5) for m = 1, ..., n_steps
    kappa[0] = 0 (no self-contribution at same time point)
    """
    kappa = np.zeros(n_steps + 1)
    m = np.arange(1, n_steps + 1)
    kappa[1:] = (m * dt) ** (H - 0.5)
    return kappa  # length n_steps+1


def _volterra_fft_batch(
    dW: np.ndarray,  # shape (n_steps, n_batch)
    kappa: np.ndarray,  # shape (n_steps+1,)
    n_steps: int,
) -> np.ndarray:
    """Compute Volterra convolution Z_{i} = sum_{j<i} kappa[i-j] dW[j] via FFT.

    Returns Z of shape (n_steps+1, n_batch) where Z[0]=0.
    """
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_steps + 1)))
    n_batch = dW.shape[1]

    # Pad kernel: we want the causal part of kappa convolved with dW
    kappa_pad = np.zeros(n_fft)
    kappa_pad[:n_steps + 1] = kappa  # kappa[0]=0, kappa[1..n_steps]

    # FFT of kernel
    K_fft = np.fft.rfft(kappa_pad)

    # Process in chunks to limit memory
    chunk = min(200, n_batch)
    Z_all = np.zeros((n_steps + 1, n_batch))

    for b_start in range(0, n_batch, chunk):
        b_end = min(b_start + chunk, n_batch)
        nb = b_end - b_start

        dW_pad = np.zeros((n_fft, nb))
        dW_pad[:n_steps, :] = dW[:, b_start:b_end]

        dW_fft = np.fft.rfft(dW_pad, axis=0)
        Z_fft = K_fft[:, None] * dW_fft  # broadcast
        Z_conv = np.fft.irfft(Z_fft, n=n_fft, axis=0)  # (n_fft, nb)

        # Z_conv[n] = sum_{j=0}^{n} kappa[n-j]*dW[j].
        # Since kappa[0]=0, the j=n term vanishes:
        #   Z_conv[n] = sum_{j=0}^{n-1} kappa[n-j]*dW[j] = Z(t_n).
        # So Z_conv[i] exactly equals the Volterra process Z(t_i) – no offset needed.
        # We need Z_all[i] = Z(t_i) for i = 0,...,n_steps.
        Z_all[:, b_start:b_end] = Z_conv[:n_steps + 1, :]

    return Z_all


def simulate_rbergomi_batch(
    n_paths: int,
    n_steps: int,
    H: float,
    nu: float,
    rho: float,
    xi0: float,
    S0: float,
    B: float,
    theta: float,
    seed: int | None = None,
    batch_size: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate rough Bergomi paths returning (S_T, S_max_full, S_max_half).

    Parameters
    ----------
    n_paths : int
    n_steps : int  (e.g. 25000)
    H : float  Hurst exponent
    nu : float  Vol-of-vol
    rho : float  Spot-vol correlation
    xi0 : float  Flat forward variance
    S0 : float  Initial spot
    B : float  Barrier
    theta : float  Maturity
    seed : int | None
    batch_size : int  Paths per FFT batch (memory control)

    Returns
    -------
    S_T : ndarray (n_paths,)  terminal stock values
    S_max_full : ndarray (n_paths,)  max over all 25000 steps
    S_max_half : ndarray (n_paths,)  max over every-2nd step
    """
    rng = np.random.default_rng(seed)
    dt = theta / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(max(0.0, 1.0 - rho**2))

    # Time grid t_i = i * dt for i = 0, ..., n_steps
    t_grid = np.arange(n_steps + 1) * dt  # shape (n_steps+1,)

    # Volterra kernel
    kappa = _build_volterra_kernel(n_steps, dt, H)
    sqrt_2H = np.sqrt(2.0 * H)

    # Variance deterministic drift: -0.5*nu^2*t_i^(2H)
    log_V_drift = -0.5 * nu**2 * t_grid ** (2.0 * H)  # shape (n_steps+1,)

    S_T = np.zeros(n_paths)
    S_max_full = np.zeros(n_paths)
    S_max_half = np.zeros(n_paths)

    start = 0
    while start < n_paths:
        end = min(start + batch_size, n_paths)
        nb = end - start

        # Generate independent Brownian increments: dW (Volterra), dW_perp (stock)
        dW = rng.standard_normal((n_steps, nb)) * sqrt_dt  # (n_steps, nb)
        dW_perp = rng.standard_normal((n_steps, nb)) * sqrt_dt  # (n_steps, nb)

        # Compute Volterra driver Z via FFT convolution
        Z = _volterra_fft_batch(dW, kappa, n_steps)  # (n_steps+1, nb)

        # Variance process: V_t = xi0 * exp(nu*sqrt(2H)*Z - 0.5*nu^2*t^(2H))
        log_V = nu * sqrt_2H * Z + log_V_drift[:, None]  # (n_steps+1, nb)
        V = xi0 * np.exp(log_V)  # (n_steps+1, nb)

        # Correlated stock Brownian:
        # dB^S = rho * dW + sqrt(1-rho^2) * dW_perp
        dB_S = rho * dW + sqrt_1_rho2 * dW_perp  # (n_steps, nb)

        # Log-price simulation (step i: use V[i], dB^S[i])
        # X_{i+1} = X_i - 0.5*V_i*dt + sqrt(V_i)*dB^S_i
        sqrt_V = np.sqrt(np.maximum(V[:-1, :], 0.0))  # (n_steps, nb)
        log_inc = -0.5 * V[:-1, :] * dt + sqrt_V * dB_S  # (n_steps, nb)
        log_price = np.zeros((n_steps + 1, nb))
        log_price[1:, :] = np.cumsum(log_inc, axis=0)
        S = S0 * np.exp(log_price)  # (n_steps+1, nb) — shape N+1

        # Running maxima (skip t=0 since S[0]=S0 < B by assumption)
        S_mx_full = S.max(axis=0)  # (nb,)
        # Half: every 2nd endpoint starting from index 2 (1-indexed step 2,4,...)
        S_mx_half = S[2::2, :].max(axis=0) if n_steps >= 2 else S_mx_full
        S_mx_half = np.maximum(S_mx_half, S0)  # ensure S0 included

        S_T[start:end] = S[-1, :]
        S_max_full[start:end] = S_mx_full
        S_max_half[start:end] = S_mx_half
        start = end

        # Free memory
        del dW, dW_perp, Z, log_V, V, dB_S, sqrt_V, log_inc, log_price, S

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
