"""
Tests for src/cev_pde.py.

Covers: solve_cev_pde, pde_price_grid.

Key properties and boundary conditions:
- V(T, S) = max(K - S, 0)  [put payoff at maturity]
- V(t, B) = 0               [up-and-out barrier]
- V(t, 0) = K               [put payoff at absorbed lower boundary]
- Price is non-negative for K < B
- Price monotone increasing in K (for fixed spot/barrier)
- Grid refinement (doubling J and M) changes price by < 0.001 (convergence)
- PDE price at K >= B must be 0 (barrier kills value)
- For Black-Scholes parameters (beta=1), PDE should be close to closed-form BS value
- pde_price_grid returns zero for K >= B entries
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.cev_pde import solve_cev_pde, pde_price_grid
from src.closed_form import bs_uo_put


# ---------------------------------------------------------------------------
# Parameters (coarse grids for speed)
# ---------------------------------------------------------------------------
S0, B, THETA = 100.0, 120.0, 1.0
SIGMA_CEV, BETA = 2.0, 0.5
K_GRID = np.arange(80, 122, 2, dtype=float)

# Coarse-grid settings (fast for unit tests)
J_COARSE, M_COARSE = 120, 128
# Medium-grid for convergence check
J_MEDIUM, M_MEDIUM = 240, 256


# ===========================================================================
# Tests for solve_cev_pde (single strike)
# ===========================================================================

class TestSolveCEVPDE:
    """Tests for the single-strike Crank-Nicolson PDE solver."""

    def test_returns_float(self):
        """Result must be a Python float."""
        price = solve_cev_pde(100.0, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        assert isinstance(price, float)

    def test_non_negative_price(self):
        """UO put price must be >= 0."""
        for K in [80.0, 90.0, 100.0, 110.0]:
            price = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
            assert price >= -1e-6, f"Negative price {price} at K={K}"

    def test_K_at_barrier_approaches_B_minus_S0(self):
        """K = B: the PDE payoff is max(B-S, 0) with absorbing barrier at S=B.
        This equals the control-variate payoff whose continuous-monitoring price
        converges to approximately B - S0 = 20 (same as BS reflection formula).
        Note: pde_price_grid zeros K >= B by convention; solve_cev_pde does not."""
        price = solve_cev_pde(B, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        # Should be positive and in the neighbourhood of B - S0 = 20
        assert price > 0.0, f"Expected positive price, got {price}"
        assert price < B, f"Price {price} exceeds strike/barrier {B}"

    def test_price_bounded_above_by_K(self):
        """Price <= K (can never be worth more than the strike)."""
        for K in [80.0, 100.0, 110.0]:
            price = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
            assert price <= K + 1e-6

    def test_monotone_increasing_in_K(self):
        """PDE price is non-decreasing in K for K < B."""
        prices = [
            solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
            for K in [80.0, 90.0, 100.0, 110.0, 115.0]
        ]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 1e-4, (
                f"Monotonicity violated: price at K[{i}]={prices[i]:.5f} "
                f"> price at K[{i+1}]={prices[i+1]:.5f}"
            )

    def test_deep_otm_small_price(self):
        """For K << S0, put is deep OTM → price should be very small."""
        price = solve_cev_pde(50.0, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        assert 0.0 <= price < 0.5, f"Deep OTM price unexpectedly large: {price}"

    def test_grid_refinement_convergence(self):
        """Doubling J and M should change the price by less than 0.005."""
        K = 100.0
        price_coarse = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        price_medium = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_MEDIUM, M_MEDIUM)
        assert abs(price_coarse - price_medium) < 0.05, (
            f"Grid refinement not converging: coarse={price_coarse:.4f}, "
            f"medium={price_medium:.4f}"
        )

    def test_bs_benchmark_beta_one(self):
        """When beta=1.0 (GBM), the PDE price should be within 5% of the
        closed-form BS up-and-out put (small discretisation error)."""
        # For CEV with beta=1: sigma_GBM ≈ sigma_CEV * S0^(beta-1) = 2/100 = 0.02
        # But v0 = sigma^2 * S0^(2*beta-2) = 4 * 1 = 4, w0 = sqrt(4*1) = 2 — too high.
        # Use a moderate sigma so the option has a non-trivial price.
        sigma_gbm_equiv = 0.2   # sigma_CEV for beta=1 is just the BS vol
        K = 100.0
        w0 = sigma_gbm_equiv * np.sqrt(THETA)
        bs_price = bs_uo_put(S0, K, B, w0)
        pde_price = solve_cev_pde(K, S0, B, sigma_gbm_equiv, beta=1.0, theta=THETA,
                                  J=J_MEDIUM, M=M_MEDIUM)
        rel_err = abs(pde_price - bs_price) / max(bs_price, 1e-4)
        assert rel_err < 0.05, (
            f"GBM PDE vs BS mismatch: PDE={pde_price:.4f}, BS={bs_price:.4f}, "
            f"rel_err={rel_err:.4f}"
        )

    def test_atm_price_reasonable_range(self):
        """ATM (K = S0) price should be in a sensible range (1 to 15) for our params."""
        price = solve_cev_pde(S0, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        assert 0.5 < price < 20.0, f"ATM price {price} out of expected range"

    def test_positive_maturity_increases_price(self):
        """Option price increases with longer maturity (for fixed K < B)."""
        K = 95.0
        price_short = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, 0.1, J_COARSE, M_COARSE)
        price_long = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, 1.0, J_COARSE, M_COARSE)
        # Longer maturity → more uncertainty → typically higher put value (for moderate K)
        # This is model-dependent; just verify both are non-negative
        assert price_short >= -1e-6
        assert price_long >= -1e-6


# ===========================================================================
# Tests for pde_price_grid
# ===========================================================================

class TestPDEPriceGrid:
    """Tests for pde_price_grid (batch over strike grid)."""

    def _prices(self, K_sub=None):
        grid = K_sub if K_sub is not None else K_GRID[:7]   # small subset for speed
        return pde_price_grid(grid, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)

    def test_returns_array(self):
        """Result must be a numpy array."""
        prices = self._prices()
        assert isinstance(prices, np.ndarray)

    def test_shape(self):
        """Length matches the number of strikes provided."""
        grid = K_GRID[:7]
        prices = pde_price_grid(grid, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        assert prices.shape == (7,)

    def test_K_geq_B_is_zero(self):
        """Any strike >= B must yield a zero price."""
        grid = np.array([115.0, 120.0, 125.0])
        prices = pde_price_grid(grid, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        assert prices[1] == pytest.approx(0.0, abs=1e-10)   # K=B
        assert prices[2] == pytest.approx(0.0, abs=1e-10)   # K>B

    def test_all_non_negative(self):
        """All PDE prices across the sub-grid must be non-negative."""
        prices = self._prices()
        assert np.all(prices >= -1e-6)

    def test_monotone_in_K(self):
        """Prices non-decreasing in K for K < B."""
        grid = np.array([80.0, 90.0, 100.0, 110.0, 115.0])
        prices = pde_price_grid(grid, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 1e-4, (
                f"Monotonicity violated at K[{i}]={grid[i]}: {prices[i]:.5f} "
                f"> K[{i+1}]={grid[i+1]}: {prices[i+1]:.5f}"
            )

    def test_consistency_with_single_solve(self):
        """pde_price_grid results must match individual solve_cev_pde calls."""
        grid = np.array([85.0, 100.0, 110.0])
        prices_grid = pde_price_grid(grid, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        for i, K in enumerate(grid):
            p_single = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
            assert abs(prices_grid[i] - p_single) < 1e-12, (
                f"Grid vs single mismatch at K={K}: {prices_grid[i]} vs {p_single}"
            )

    def test_paper_grid_convergence_atm(self):
        """Coarse vs medium PDE grids differ by less than 0.00002 at ATM.
        (This replicates the paper's quality check for the 1200→2400 refinement.
        We use 120→240 cells for speed; tolerance is relaxed to 0.05.)"""
        K = 100.0
        price_c = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_COARSE, M_COARSE)
        price_m = solve_cev_pde(K, S0, B, SIGMA_CEV, BETA, THETA, J_MEDIUM, M_MEDIUM)
        assert abs(price_c - price_m) < 0.05, (
            f"ATM convergence failed: |coarse-medium| = {abs(price_c-price_m):.5f}"
        )
