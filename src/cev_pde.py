"""
CEV barrier option PDE solver (Crank-Nicolson).

Solves the continuous-monitoring up-and-out put PDE for the CEV model:
  partial_t V + 0.5 * sigma^2 * S^(2*beta) * partial_{SS} V = 0
  V(T, S) = max(K - S, 0) for S in [0, B)
  V(t, B) = 0  (up-and-out)
  V(t, 0) = K  (degenerate boundary for put, absorbed at S=0)

Time marches backward from T to 0.
Spatial domain: [0, B] with uniform grid of J+1 nodes.
Crank-Nicolson with 4 initial backward-Euler half steps for startup damping.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Tuple


def solve_cev_pde(
    K: float,
    S0: float,
    B: float,
    sigma: float,
    beta: float,
    theta: float,
    J: int,
    M: int,
) -> float:
    """Solve CEV up-and-out put PDE and return price at S0.

    Parameters
    ----------
    K : float  Strike
    S0 : float  Spot price
    B : float  Barrier
    sigma : float  CEV diffusion
    beta : float  CEV exponent
    theta : float  Maturity
    J : int  Number of spatial cells (J+1 nodes: S_0=0 … S_J=B)
    M : int  Number of time steps

    Returns
    -------
    float: option price at S0
    """
    dS = B / J
    S_grid = np.linspace(0.0, B, J + 1)   # shape (J+1,)
    dt = theta / M

    # Interior nodes: indices 1 .. J-1 in S_grid
    n_int = J - 1
    S_int = S_grid[1:J]                    # shape (n_int,)

    # Diffusion coefficient at interior nodes
    D = 0.5 * sigma ** 2 * S_int ** (2.0 * beta)

    # FD coefficients for the second-derivative operator L:
    #   L_j V = a_j*(V_{j-1} - 2*V_j + V_{j+1}) / dS^2
    a_diag = D / dS ** 2     # shape (n_int,)   coupling to each neighbour
    b_diag = -2.0 * D / dS ** 2  # shape (n_int,) self-coupling (negative)

    def _build(theta_cn: float, dt_use: float) -> sp.csc_matrix:
        """Build (I - theta_cn*dt_use*L_int) as a sparse tridiagonal matrix.

        For the implicit CN part  : theta_cn =  0.5  → (I - 0.5*dt*L)
        For the explicit CN part  : theta_cn = -0.5  → (I + 0.5*dt*L)
        For a full backward-Euler : theta_cn =  1.0  → (I - dt_use*L)
        """
        main  = 1.0 - theta_cn * dt_use * b_diag
        upper = -theta_cn * dt_use * a_diag[:-1]   # super-diagonal, length n_int-1
        lower = -theta_cn * dt_use * a_diag[1:]    # sub-diagonal,   length n_int-1
        return sp.diags(
            [lower, main, upper], [-1, 0, 1],
            shape=(n_int, n_int), format="csc",
        )

    # Pre-build matrices used in the main CN loop (built once, reused M-2 times)
    A_imp = _build(0.5, dt)   # (I - 0.5*dt*L)  – implicit
    A_exp = _build(-0.5, dt)  # (I + 0.5*dt*L)  – explicit

    # Boundary conditions (time-independent)
    bc_left  = K    # V(t, 0) = K  (intrinsic put at absorbed boundary)
    bc_right = 0.0  # V(t, B) = 0  (barrier)

    # Initial condition at maturity (backward time starts here)
    V_int = np.maximum(K - S_int, 0.0)

    def _step_be(v: np.ndarray, dt_use: float) -> np.ndarray:
        """One backward-Euler step of size dt_use."""
        A = _build(1.0, dt_use)
        rhs = v.copy()
        rhs[0]  += dt_use * a_diag[0]  * bc_left
        rhs[-1] += dt_use * a_diag[-1] * bc_right
        return spsolve(A, rhs)

    def _step_cn(v: np.ndarray) -> np.ndarray:
        """One Crank-Nicolson step of size dt."""
        rhs = A_exp @ v
        # Boundary contributions: explicit (0.5*dt) + implicit (0.5*dt) = dt total
        rhs[0]  += dt * a_diag[0]  * bc_left
        rhs[-1] += dt * a_diag[-1] * bc_right
        return spsolve(A_imp, rhs)

    # 4 backward-Euler half-steps for startup damping (= 2 full dt worth of time)
    for _ in range(4):
        V_int = _step_be(V_int, dt / 2.0)

    # Remaining M-2 Crank-Nicolson steps
    for _ in range(M - 2):
        V_int = _step_cn(V_int)

    # Reconstruct full solution and interpolate at S0
    V_full = np.empty(J + 1)
    V_full[0]  = bc_left
    V_full[-1] = bc_right
    V_full[1:J] = V_int

    idx = int(np.floor(S0 / dS))
    idx = int(np.clip(idx, 0, J - 1))
    frac = (S0 - S_grid[idx]) / dS
    return float((1.0 - frac) * V_full[idx] + frac * V_full[idx + 1])


def pde_price_grid(
    K_grid: np.ndarray,
    S0: float,
    B: float,
    sigma: float,
    beta: float,
    theta: float,
    J: int,
    M: int,
) -> np.ndarray:
    """Compute PDE prices for a full strike grid.

    Parameters
    ----------
    K_grid : ndarray  Strike prices
    S0, B, sigma, beta, theta : model parameters
    J : int  Number of spatial cells
    M : int  Number of time steps

    Returns
    -------
    prices : ndarray (n_strikes,)
    """
    prices = np.zeros(len(K_grid))
    for i, K in enumerate(K_grid):
        if K >= B:
            prices[i] = 0.0
        else:
            prices[i] = solve_cev_pde(K, S0, B, sigma, beta, theta, J, M)
    return prices
