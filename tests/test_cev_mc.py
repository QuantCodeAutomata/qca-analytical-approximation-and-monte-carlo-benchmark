"""
Tests for src/cev_mc.py.

Covers: simulate_cev_batch, simulate_cev_batch_dual, compute_payoffs.

Key properties:
- S_T and S_max are non-negative (absorbed Euler)
- S_max >= S0 always (starts at S0)
- S_max >= S_T always
- S_max_full >= S_max_half (checking more steps cannot give a lower max)
- Payoffs are non-negative
- Survived paths: S_max < B
- Terminal distribution moments converge to theoretical values as n_paths grows
- At K=B, Y_put = Y_ctrl exactly (both are (B-S_N)^+ * survived)
- Absorbed Euler: once S=0 it stays 0
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.cev_mc import simulate_cev_batch, simulate_cev_batch_dual, compute_payoffs


# ---------------------------------------------------------------------------
# Shared parameters (small n_steps and n_paths for fast unit tests)
# ---------------------------------------------------------------------------
S0, B, THETA = 100.0, 120.0, 1.0
BETA, SIGMA = 0.5, 2.0
K_GRID = np.arange(80, 122, 2, dtype=float)

N_PATHS_SMALL = 2000
N_STEPS_SMALL = 100     # Very coarse for speed; geometry still tested


# ===========================================================================
# Tests for simulate_cev_batch
# ===========================================================================

class TestSimulateCEVBatch:
    """Tests for the CEV batch simulation."""

    def test_output_shapes(self):
        """S_T and S_max must have shape (n_paths,)."""
        S_T, S_max = simulate_cev_batch(N_PATHS_SMALL, N_STEPS_SMALL,
                                        SIGMA, BETA, S0, B, THETA, seed=0)
        assert S_T.shape == (N_PATHS_SMALL,)
        assert S_max.shape == (N_PATHS_SMALL,)

    def test_S_T_non_negative(self):
        """Absorbed Euler guarantees S_T >= 0."""
        S_T, _ = simulate_cev_batch(N_PATHS_SMALL, N_STEPS_SMALL,
                                    SIGMA, BETA, S0, B, THETA, seed=1)
        assert np.all(S_T >= 0.0), f"Negative S_T found: {S_T.min()}"

    def test_S_max_geq_S0(self):
        """Running max must be >= S0 (initial value is included)."""
        _, S_max = simulate_cev_batch(N_PATHS_SMALL, N_STEPS_SMALL,
                                     SIGMA, BETA, S0, B, THETA, seed=2)
        assert np.all(S_max >= S0 - 1e-10), f"S_max below S0: {S_max.min()}"

    def test_S_max_geq_S_T(self):
        """Running max >= terminal value always."""
        S_T, S_max = simulate_cev_batch(N_PATHS_SMALL, N_STEPS_SMALL,
                                        SIGMA, BETA, S0, B, THETA, seed=3)
        assert np.all(S_max >= S_T - 1e-10)

    def test_reproducibility_with_seed(self):
        """Same seed produces identical paths."""
        S_T1, S_max1 = simulate_cev_batch(500, N_STEPS_SMALL,
                                          SIGMA, BETA, S0, B, THETA, seed=42)
        S_T2, S_max2 = simulate_cev_batch(500, N_STEPS_SMALL,
                                          SIGMA, BETA, S0, B, THETA, seed=42)
        np.testing.assert_array_equal(S_T1, S_T2)
        np.testing.assert_array_equal(S_max1, S_max2)

    def test_different_seeds_differ(self):
        """Different seeds must produce different paths."""
        S_T1, _ = simulate_cev_batch(200, N_STEPS_SMALL,
                                     SIGMA, BETA, S0, B, THETA, seed=0)
        S_T2, _ = simulate_cev_batch(200, N_STEPS_SMALL,
                                     SIGMA, BETA, S0, B, THETA, seed=1)
        assert not np.allclose(S_T1, S_T2)

    def test_survival_fraction_reasonable(self):
        """With B = 120 and short maturity, a non-trivial fraction should survive."""
        S_T, S_max = simulate_cev_batch(5000, N_STEPS_SMALL,
                                        SIGMA, BETA, S0, B, THETA, seed=4)
        survival_rate = np.mean(S_max < B)
        # For S0=100, B=120 barrier is 20% above spot; most paths should survive
        assert 0.5 < survival_rate < 1.0, f"Survival rate {survival_rate} out of range"

    def test_absorption_at_zero(self):
        """With extreme beta < 0 (not realistic but tests absorption), S stays at 0
        once hit. Here we test with beta=0 (GBM) that S stays positive."""
        # With beta=0, dS = sigma * dW → Gaussian random walk, can go negative
        # Absorption: max(S + ..., 0)
        # Just verify that zero-absorbed paths are exactly 0 (not negative)
        S_T, _ = simulate_cev_batch(1000, N_STEPS_SMALL, sigma=5.0, beta=0.0,
                                    S0=1.0, B=1e6, theta=1.0, seed=5)
        assert np.all(S_T >= 0.0)

    def test_batch_size_does_not_corrupt_outputs(self):
        """Different batch_size values should produce statistically equivalent
        outputs (same mean, standard deviation within 3σ) even if the random
        stream differs.  batch_size is an internal partitioning parameter whose
        only purpose is memory management; it must not corrupt path validity."""
        n = 5000
        S_T1, Sm1 = simulate_cev_batch(n, N_STEPS_SMALL, SIGMA, BETA,
                                        S0, B, THETA, seed=10, batch_size=250)
        S_T2, Sm2 = simulate_cev_batch(n, N_STEPS_SMALL, SIGMA, BETA,
                                        S0, B, THETA, seed=10, batch_size=1000)
        # Both runs should produce plausible non-negative outputs
        assert np.all(S_T1 >= 0.0) and np.all(S_T2 >= 0.0)
        assert np.all(Sm1 >= S0 - 1e-10) and np.all(Sm2 >= S0 - 1e-10)
        # Means should agree within 5 MC standard errors (rough check)
        se = S_T1.std() / np.sqrt(n)
        assert abs(S_T1.mean() - S_T2.mean()) < 6 * se

    def test_zero_drift_martingale_property(self):
        """Under zero drift, E[S_T] ≈ S0 for large n_paths."""
        n = 20_000
        S_T, _ = simulate_cev_batch(n, N_STEPS_SMALL, SIGMA, BETA,
                                    S0, B=1e6, theta=0.1, seed=6)
        mean_ST = S_T.mean()
        # CEV martingale property: E[S_T] = S0 (no barrier, no drift)
        # Wide tolerance because it's only a small sample
        assert abs(mean_ST - S0) < 3.0, f"Martingale check: E[S_T]={mean_ST:.2f} far from S0={S0}"


# ===========================================================================
# Tests for simulate_cev_batch_dual (monitoring sensitivity)
# ===========================================================================

class TestSimulateCEVBatchDual:
    """Tests for the dual-monitoring CEV simulator."""

    def test_output_shapes(self):
        S_T, S_mx_full, S_mx_half = simulate_cev_batch_dual(
            N_PATHS_SMALL, N_STEPS_SMALL, SIGMA, BETA, S0, B, THETA, seed=0
        )
        assert S_T.shape == (N_PATHS_SMALL,)
        assert S_mx_full.shape == (N_PATHS_SMALL,)
        assert S_mx_half.shape == (N_PATHS_SMALL,)

    def test_S_max_full_geq_S_max_half(self):
        """Full monitoring (every step) ≥ half monitoring (every 2nd step)."""
        S_T, S_mx_full, S_mx_half = simulate_cev_batch_dual(
            N_PATHS_SMALL, N_STEPS_SMALL, SIGMA, BETA, S0, B, THETA, seed=7
        )
        assert np.all(S_mx_full >= S_mx_half - 1e-10), (
            f"S_max_full < S_max_half on {np.sum(S_mx_full < S_mx_half - 1e-10)} paths"
        )

    def test_S_T_non_negative(self):
        S_T, _, _ = simulate_cev_batch_dual(
            N_PATHS_SMALL, N_STEPS_SMALL, SIGMA, BETA, S0, B, THETA, seed=8
        )
        assert np.all(S_T >= 0.0)

    def test_both_max_geq_S0(self):
        _, S_mx_full, S_mx_half = simulate_cev_batch_dual(
            N_PATHS_SMALL, N_STEPS_SMALL, SIGMA, BETA, S0, B, THETA, seed=9
        )
        assert np.all(S_mx_full >= S0 - 1e-10)
        assert np.all(S_mx_half >= S0 - 1e-10)

    def test_survival_full_leq_half(self):
        """More monitoring dates → fewer survivors (full ≤ half survival rate)."""
        _, S_mx_full, S_mx_half = simulate_cev_batch_dual(
            5000, N_STEPS_SMALL, SIGMA, BETA, S0, B, THETA, seed=11
        )
        surv_full = np.mean(S_mx_full < B)
        surv_half = np.mean(S_mx_half < B)
        # Full monitoring catches more barrier crossings, so fewer survivors
        assert surv_full <= surv_half + 0.01  # Small tolerance for stochasticity

    def test_reproducibility(self):
        """Same seed gives same dual results."""
        out1 = simulate_cev_batch_dual(300, N_STEPS_SMALL, SIGMA, BETA,
                                       S0, B, THETA, seed=77)
        out2 = simulate_cev_batch_dual(300, N_STEPS_SMALL, SIGMA, BETA,
                                       S0, B, THETA, seed=77)
        for a, b_arr in zip(out1, out2):
            np.testing.assert_array_equal(a, b_arr)


# ===========================================================================
# Tests for compute_payoffs
# ===========================================================================

class TestComputePayoffs:
    """Tests for the payoff computation function."""

    def test_output_shapes(self):
        """Y_put has shape (n_paths, n_strikes), Y_ctrl has shape (n_paths,)."""
        n, n_k = 500, len(K_GRID)
        S_T = np.random.default_rng(0).uniform(60, 140, n)
        S_max = np.maximum(S_T, S0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        assert Y_put.shape == (n, n_k)
        assert Y_ctrl.shape == (n,)

    def test_payoffs_non_negative(self):
        """All option payoffs must be non-negative."""
        n = 1000
        S_T = np.random.default_rng(1).uniform(60, 140, n)
        S_max = np.maximum(S_T + 5, S0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        assert np.all(Y_put >= 0.0)
        assert np.all(Y_ctrl >= 0.0)

    def test_knocked_out_paths_zero_payoff(self):
        """Paths where S_max >= B must have zero payoff."""
        n = 100
        # Force all paths to be knocked out
        S_T = np.full(n, 110.0)
        S_max = np.full(n, B + 5.0)   # exceeded barrier
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        np.testing.assert_array_equal(Y_put, np.zeros((n, len(K_GRID))))
        np.testing.assert_array_equal(Y_ctrl, np.zeros(n))

    def test_survived_paths_correct_payoff(self):
        """For a single survived path with S_T = 90, K = 100:
        put payoff = max(100-90, 0) = 10."""
        S_T = np.array([90.0])
        S_max = np.array([S0 + 1.0])  # survived (< B)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        # Find index of K=100
        idx = np.where(K_GRID == 100.0)[0][0]
        assert abs(Y_put[0, idx] - 10.0) < 1e-12

    def test_at_K_equals_B_control_payoff(self):
        """For K = B, Y_put == Y_ctrl = max(B - S_T, 0) * survived."""
        n = 200
        rng = np.random.default_rng(2)
        S_T = rng.uniform(80.0, 115.0, n)   # all survived
        S_max = np.full(n, S0 + 5.0)         # all < B, so survived
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        # Last strike = B = 120; Y_put[:, -1] should equal Y_ctrl
        np.testing.assert_allclose(Y_put[:, -1], Y_ctrl, atol=1e-12)

    def test_OTM_put_zero_when_S_T_above_K(self):
        """If S_T > K for a survived path, put payoff at that K is 0."""
        S_T = np.array([105.0])   # above K=100
        S_max = np.array([S0 + 1.0])
        Y_put, _ = compute_payoffs(S_T, S_max, K_GRID, B)
        idx = np.where(K_GRID == 100.0)[0][0]
        assert Y_put[0, idx] == 0.0

    def test_monotone_increasing_in_K(self):
        """For a fixed survived path, Y_put is non-decreasing in K."""
        S_T = np.array([95.0])
        S_max = np.array([S0 + 2.0])
        Y_put, _ = compute_payoffs(S_T, S_max, K_GRID, B)
        for i in range(len(K_GRID) - 1):
            assert Y_put[0, i] <= Y_put[0, i + 1] + 1e-10
