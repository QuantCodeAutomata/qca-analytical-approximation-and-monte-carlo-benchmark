"""
Tests for src/sabr_mc.py.

Covers: simulate_sabr_batch, compute_payoffs.

Key properties:
- S_max_full >= S_max_half (more monitoring dates ≥ running max)
- S_T unrestricted (Euler allows slight negatives but very rarely for ρ=-0.9)
- S_max_full >= S0 (initial value included in running max)
- Survival fraction reasonable for B=120, S0=100
- Payoffs non-negative
- Sigma remains positive (lognormal exact step)
- Identical seeds → identical paths
- Lognormal vol step: sigma_i+1 = sigma_i * exp(nu*dB - 0.5*nu^2*dt) > 0 always
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.sabr_mc import simulate_sabr_batch, compute_payoffs


# ---------------------------------------------------------------------------
# Parameters (coarse settings for fast unit tests)
# ---------------------------------------------------------------------------
S0, B, THETA = 100.0, 120.0, 1.0
NU, RHO, SIGMA0 = 0.2, -0.9, 0.2
K_GRID = np.arange(80, 122, 2, dtype=float)

N_PATHS = 2000
N_STEPS = 200   # coarse for speed; geometry still tested


# ===========================================================================
# Tests for simulate_sabr_batch
# ===========================================================================

class TestSimulateSABRBatch:
    """Tests for the lognormal-SABR batch simulator."""

    def _sim(self, n=N_PATHS, seed=0):
        return simulate_sabr_batch(n, N_STEPS, NU, RHO, SIGMA0, S0, B, THETA, seed=seed)

    def test_output_shapes(self):
        S_T, S_mx_full, S_mx_half = self._sim()
        assert S_T.shape == (N_PATHS,)
        assert S_mx_full.shape == (N_PATHS,)
        assert S_mx_half.shape == (N_PATHS,)

    def test_S_max_full_geq_S0(self):
        """Full running max must be >= S0."""
        _, S_mx_full, _ = self._sim(seed=1)
        assert np.all(S_mx_full >= S0 - 1e-10)

    def test_S_max_half_geq_S0(self):
        """Half running max must be >= S0."""
        _, _, S_mx_half = self._sim(seed=2)
        assert np.all(S_mx_half >= S0 - 1e-10)

    def test_S_max_full_geq_half(self):
        """Full monitoring max >= half monitoring max (more dates = larger or equal max)."""
        _, S_mx_full, S_mx_half = self._sim(seed=3)
        assert np.all(S_mx_full >= S_mx_half - 1e-10)

    def test_reproducibility(self):
        """Identical seeds must produce identical results."""
        out1 = simulate_sabr_batch(500, N_STEPS, NU, RHO, SIGMA0, S0, B, THETA, seed=42)
        out2 = simulate_sabr_batch(500, N_STEPS, NU, RHO, SIGMA0, S0, B, THETA, seed=42)
        for a, b_arr in zip(out1, out2):
            np.testing.assert_array_equal(a, b_arr)

    def test_different_seeds_differ(self):
        """Different seeds produce different terminal values."""
        S_T1, _, _ = self._sim(seed=10)
        S_T2, _, _ = self._sim(seed=11)
        assert not np.allclose(S_T1, S_T2)

    def test_survival_fraction_reasonable(self):
        """With B=120%, S0=100, most paths should survive (barrier is 20% OTM)."""
        _, S_mx_full, _ = self._sim(n=5000, seed=4)
        surv = np.mean(S_mx_full < B)
        assert 0.5 < surv < 1.0, f"Survival rate {surv} out of expected range"

    def test_full_survival_leq_half_survival(self):
        """More monitoring → at most as many survivors as less monitoring."""
        _, S_mx_full, S_mx_half = self._sim(n=5000, seed=5)
        surv_full = np.mean(S_mx_full < B)
        surv_half = np.mean(S_mx_half < B)
        assert surv_full <= surv_half + 0.02  # small tolerance for randomness

    def test_vol_always_positive(self):
        """Because vol uses exact lognormal step, sigma_t > 0 always.
        Proxy: use tiny nu and verify S is well-behaved."""
        # With tiny nu, SABR ≈ GBM; S should stay positive (though Euler can give
        # S < 0 for large sigma0; with sigma0=0.2 and short maturity it's fine)
        S_T, _, _ = simulate_sabr_batch(
            1000, N_STEPS, nu=0.001, rho=0.0, sigma0=0.2,
            S0=S0, B=1e6, theta=0.1, seed=6,
        )
        # For very low nu and short theta, virtually all S_T should be positive
        frac_positive = np.mean(S_T > 0)
        assert frac_positive > 0.98, f"Too many negative S_T: {frac_positive:.3f}"

    def test_high_rho_shifts_distribution(self):
        """Positive rho should make crashes rarer (S and sigma move together),
        compared to negative rho.  Proxy: survival rate under positive rho is lower
        because S tends to rise when sigma rises → more barrier hits for an up barrier."""
        # With rho=+0.9, higher vol → higher S → more barrier hits
        _, mx_pos, _ = simulate_sabr_batch(5000, N_STEPS, NU, +0.9, SIGMA0, S0, B, THETA, seed=7)
        _, mx_neg, _ = simulate_sabr_batch(5000, N_STEPS, NU, -0.9, SIGMA0, S0, B, THETA, seed=7)
        surv_pos = np.mean(mx_pos < B)
        surv_neg = np.mean(mx_neg < B)
        # Positive rho → more paths hit the up barrier → lower survival
        assert surv_pos < surv_neg + 0.05


# ===========================================================================
# Tests for compute_payoffs (same function as CEV, re-verified here)
# ===========================================================================

class TestSABRComputePayoffs:
    """Tests for the SABR payoff computation."""

    def test_shapes(self):
        n, n_k = 300, len(K_GRID)
        S_T = np.random.default_rng(0).uniform(70, 115, n)
        S_max = np.full(n, S0 + 5.0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        assert Y_put.shape == (n, n_k)
        assert Y_ctrl.shape == (n,)

    def test_knocked_out_all_zero(self):
        """If all S_max >= B, all payoffs are zero."""
        n = 50
        S_T = np.full(n, 95.0)
        S_max = np.full(n, B)   # exactly at barrier → knocked out
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        np.testing.assert_array_equal(Y_put, np.zeros((n, len(K_GRID))))
        np.testing.assert_array_equal(Y_ctrl, np.zeros(n))

    def test_survived_put_payoff_correct(self):
        """For S_T=90, K=100, survived: payoff = 10."""
        S_T = np.array([90.0])
        S_max = np.array([S0 + 2.0])  # < B: survived
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        idx = np.where(K_GRID == 100.0)[0][0]
        assert abs(Y_put[0, idx] - 10.0) < 1e-12

    def test_payoffs_non_negative(self):
        n = 200
        S_T = np.random.default_rng(1).uniform(70, 130, n)
        S_max = np.maximum(S_T, S0 + 1.0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        assert np.all(Y_put >= 0.0)
        assert np.all(Y_ctrl >= 0.0)

    def test_K_equals_B_matches_ctrl(self):
        """At K=B, Y_put[:, -1] = Y_ctrl."""
        n = 100
        rng = np.random.default_rng(2)
        S_T = rng.uniform(80, 115, n)
        S_max = np.full(n, S0 + 5.0)   # survived
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        np.testing.assert_allclose(Y_put[:, -1], Y_ctrl, atol=1e-12)

    def test_monotone_increasing_in_K(self):
        """Put payoff is non-decreasing in K for a fixed survived path."""
        S_T = np.array([92.0])
        S_max = np.array([S0 + 3.0])
        Y_put, _ = compute_payoffs(S_T, S_max, K_GRID, B)
        for i in range(len(K_GRID) - 1):
            assert Y_put[0, i] <= Y_put[0, i + 1] + 1e-10
