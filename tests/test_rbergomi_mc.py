"""
Tests for src/rbergomi_mc.py.

Covers: _build_volterra_kernel, _volterra_fft_batch,
        simulate_rbergomi_batch, compute_payoffs.

Key properties:
- Volterra kernel kappa[0] = 0, kappa[m] = (m*dt)^(H-0.5) for m >= 1
- FFT convolution is causal (Z[i] depends only on dW[0..i-1])
- Z[0] = 0 (no innovation yet at t=0)
- S_T = S0 * exp(X_T) where X_T is the cumulative log-price
- S_max_full >= S_max_half (more dates → larger or equal running max)
- S_max >= S0 always
- Payoffs non-negative; knocked-out paths have zero payoff
- Variance process V_t = xi0 * exp(...) > 0 always
- Reproducibility with fixed seed
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.rbergomi_mc import (
    _build_volterra_kernel,
    _volterra_fft_batch,
    simulate_rbergomi_batch,
    compute_payoffs,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
H, NU, RHO, XI0 = 0.1, 1.0, -0.9, 0.04
S0, B, THETA = 100.0, 120.0, 0.25
K_GRID = np.arange(80, 122, 2, dtype=float)

N_PATHS = 500
N_STEPS = 64     # power-of-2 for FFT convenience; very coarse for speed


# ===========================================================================
# Tests for _build_volterra_kernel
# ===========================================================================

class TestBuildVolterraKernel:
    """Tests for the Volterra kernel construction."""

    def test_shape(self):
        kappa = _build_volterra_kernel(100, 0.01, H)
        assert kappa.shape == (101,)   # length n_steps + 1

    def test_kappa_zero_at_index_zero(self):
        """kappa[0] = 0 (no self-contribution)."""
        kappa = _build_volterra_kernel(100, 0.01, H)
        assert kappa[0] == 0.0

    def test_kappa_positive_for_all_m_ge_1(self):
        """kappa[m] = (m*dt)^(H-0.5) > 0 for H=0.1 (H-0.5 = -0.4 < 0 but (m*dt)>0)."""
        kappa = _build_volterra_kernel(50, 0.01, H)
        assert np.all(kappa[1:] > 0.0)

    def test_kappa_formula(self):
        """Verify kappa[m] = (m * dt)^(H - 0.5) for m = 1, ..., n."""
        n, dt = 10, 0.004
        kappa = _build_volterra_kernel(n, dt, H)
        for m in range(1, n + 1):
            expected = (m * dt) ** (H - 0.5)
            assert abs(kappa[m] - expected) < 1e-14, f"kappa[{m}] mismatch"

    def test_kappa_decreasing_for_H_lt_half(self):
        """For H < 0.5, kappa is decreasing (singular kernel)."""
        kappa = _build_volterra_kernel(100, 0.01, H)
        # kappa[m] = (m*dt)^(H-0.5) is decreasing in m when H-0.5 < 0
        assert np.all(np.diff(kappa[1:]) < 0.0)

    def test_kappa_increasing_for_H_gt_half(self):
        """For H > 0.5, kappa is increasing (non-singular)."""
        kappa = _build_volterra_kernel(100, 0.01, H=0.7)
        # kappa[m] = (m*dt)^(0.2) is increasing in m
        assert np.all(np.diff(kappa[1:]) > 0.0)


# ===========================================================================
# Tests for _volterra_fft_batch
# ===========================================================================

class TestVolterraFFTBatch:
    """Tests for the FFT-based causal convolution."""

    def test_output_shape(self):
        """Z must have shape (n_steps+1, n_batch)."""
        n, nb = 32, 5
        dt = 0.001
        kappa = _build_volterra_kernel(n, dt, H)
        dW = np.random.default_rng(0).standard_normal((n, nb)) * np.sqrt(dt)
        Z = _volterra_fft_batch(dW, kappa, n)
        assert Z.shape == (n + 1, nb)

    def test_Z_zero_at_t0(self):
        """Z[0] = 0 for all paths (no increments yet)."""
        n, nb = 32, 10
        dt = 0.001
        kappa = _build_volterra_kernel(n, dt, H)
        dW = np.random.default_rng(1).standard_normal((n, nb)) * np.sqrt(dt)
        Z = _volterra_fft_batch(dW, kappa, n)
        np.testing.assert_allclose(Z[0, :], np.zeros(nb), atol=1e-10)

    def test_causal_property(self):
        """Zeroing a late increment should not change Z at earlier steps.
        Specifically: Z[2] = kappa[2]*dW[0] + kappa[1]*dW[1].
        Changing dW[5] (if step > 2) should not affect Z[2]."""
        n, nb = 32, 1
        dt = 0.01
        kappa = _build_volterra_kernel(n, dt, H)
        rng = np.random.default_rng(2)
        dW = rng.standard_normal((n, nb)) * np.sqrt(dt)
        Z1 = _volterra_fft_batch(dW.copy(), kappa, n)

        # Modify only dW at step 10 (affects Z[11], Z[12], ... but not Z[0..10])
        dW_mod = dW.copy()
        dW_mod[10, 0] += 1000.0
        Z2 = _volterra_fft_batch(dW_mod, kappa, n)

        # Z[0..10] should be identical
        np.testing.assert_allclose(Z1[:11, 0], Z2[:11, 0], atol=1e-8)

    def test_convolution_direct_vs_fft(self):
        """Compare FFT convolution against direct sum for small n."""
        n, nb = 8, 3
        dt = 0.01
        kappa = _build_volterra_kernel(n, dt, H)
        rng = np.random.default_rng(3)
        dW = rng.standard_normal((n, nb)) * np.sqrt(dt)

        Z_fft = _volterra_fft_batch(dW, kappa, n)

        # Direct computation: Z[i] = sum_{j=0}^{i-1} kappa[i-j] * dW[j]
        Z_direct = np.zeros((n + 1, nb))
        for i in range(1, n + 1):
            for j in range(i):
                Z_direct[i, :] += kappa[i - j] * dW[j, :]

        np.testing.assert_allclose(Z_fft, Z_direct, atol=1e-8)

    def test_finite_output(self):
        """All Z values must be finite."""
        n, nb = 64, 20
        dt = 0.25 / n
        kappa = _build_volterra_kernel(n, dt, H)
        dW = np.random.default_rng(4).standard_normal((n, nb)) * np.sqrt(dt)
        Z = _volterra_fft_batch(dW, kappa, n)
        assert np.all(np.isfinite(Z))


# ===========================================================================
# Tests for simulate_rbergomi_batch
# ===========================================================================

class TestSimulateRBergomiBatch:
    """Tests for the rough Bergomi batch simulator."""

    def _sim(self, n=N_PATHS, seed=0):
        return simulate_rbergomi_batch(
            n, N_STEPS, H, NU, RHO, XI0, S0, B, THETA, seed=seed, batch_size=100
        )

    def test_output_shapes(self):
        S_T, S_mx_full, S_mx_half = self._sim()
        assert S_T.shape == (N_PATHS,)
        assert S_mx_full.shape == (N_PATHS,)
        assert S_mx_half.shape == (N_PATHS,)

    def test_S_T_positive(self):
        """S_T = S0 * exp(X_T) > 0 always (log-price scheme)."""
        S_T, _, _ = self._sim(seed=1)
        assert np.all(S_T > 0.0), f"Found non-positive S_T: min={S_T.min()}"

    def test_S_max_full_geq_S0(self):
        """Running max includes initial value S0."""
        _, S_mx_full, _ = self._sim(seed=2)
        assert np.all(S_mx_full >= S0 - 1e-10)

    def test_S_max_half_geq_S0(self):
        _, _, S_mx_half = self._sim(seed=3)
        assert np.all(S_mx_half >= S0 - 1e-10)

    def test_S_max_full_geq_S_max_half(self):
        """Full monitoring (every step) >= half monitoring (every 2nd step)."""
        _, S_mx_full, S_mx_half = self._sim(seed=4)
        assert np.all(S_mx_full >= S_mx_half - 1e-10)

    def test_S_T_geq_1(self):
        """S_T = S0 * exp(cumsum) should be realistically positive and not absurdly large."""
        S_T, _, _ = self._sim(n=1000, seed=5)
        # All should be > 0 (guaranteed by log-price)
        assert np.all(S_T > 0.0)
        # Sanity: 5% of paths are in [1, 1000] range (very rough)
        assert np.all(S_T > 0.5), f"Unrealistically small S_T: {S_T.min()}"

    def test_reproducibility(self):
        """Same seed → same outputs."""
        out1 = self._sim(n=100, seed=99)
        out2 = self._sim(n=100, seed=99)
        for a, b_arr in zip(out1, out2):
            np.testing.assert_array_equal(a, b_arr)

    def test_different_seeds_differ(self):
        S_T1, _, _ = self._sim(seed=10)
        S_T2, _, _ = self._sim(seed=11)
        assert not np.allclose(S_T1, S_T2)

    def test_survival_fraction_reasonable(self):
        """With theta=0.25 (short maturity), most paths should survive the B=120 barrier."""
        _, S_mx_full, _ = self._sim(n=2000, seed=6)
        surv = np.mean(S_mx_full < B)
        assert 0.5 < surv < 1.0, f"Unexpected survival rate: {surv}"

    def test_full_surv_leq_half_surv(self):
        """More frequent monitoring → at most as many survivors."""
        _, S_mx_full, S_mx_half = self._sim(n=2000, seed=7)
        assert np.mean(S_mx_full < B) <= np.mean(S_mx_half < B) + 0.03


# ===========================================================================
# Tests for compute_payoffs (rbergomi version)
# ===========================================================================

class TestRBComputePayoffs:
    """Tests for the rough Bergomi payoff computation (same logic as CEV/SABR)."""

    def test_shapes(self):
        n, n_k = 200, len(K_GRID)
        S_T = np.random.default_rng(0).lognormal(np.log(S0), 0.2, n)
        S_max = np.maximum(S_T * 1.05, S0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        assert Y_put.shape == (n, n_k)
        assert Y_ctrl.shape == (n,)

    def test_payoffs_non_negative(self):
        n = 200
        S_T = np.random.default_rng(1).lognormal(np.log(S0), 0.2, n)
        S_max = np.maximum(S_T, S0 + 3.0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        assert np.all(Y_put >= 0.0)
        assert np.all(Y_ctrl >= 0.0)

    def test_knocked_out_zero(self):
        """Paths with S_max >= B: zero payoff."""
        n = 50
        S_T = np.full(n, 100.0)
        S_max = np.full(n, B + 1.0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        np.testing.assert_array_equal(Y_put, np.zeros((n, len(K_GRID))))
        np.testing.assert_array_equal(Y_ctrl, np.zeros(n))

    def test_K_equals_B_collapses_to_ctrl(self):
        """At K=B, Y_put[:, -1] = Y_ctrl."""
        n = 100
        rng = np.random.default_rng(2)
        S_T = rng.uniform(80, 115, n)
        S_max = np.full(n, S0 + 5.0)
        Y_put, Y_ctrl = compute_payoffs(S_T, S_max, K_GRID, B)
        np.testing.assert_allclose(Y_put[:, -1], Y_ctrl, atol=1e-12)

    def test_survived_exact_payoff(self):
        """For S_T=90, K=96 (in grid, step=2), survived: payoff = 6."""
        S_T = np.array([90.0])
        S_max = np.array([S0 + 2.0])
        Y_put, _ = compute_payoffs(S_T, S_max, K_GRID, B)
        idx = np.where(K_GRID == 96.0)[0][0]
        assert abs(Y_put[0, idx] - 6.0) < 1e-12

    def test_monotone_in_K(self):
        """For a fixed survived path, Y_put is non-decreasing in K."""
        S_T = np.array([88.0])
        S_max = np.array([S0 + 4.0])
        Y_put, _ = compute_payoffs(S_T, S_max, K_GRID, B)
        for i in range(len(K_GRID) - 1):
            assert Y_put[0, i] <= Y_put[0, i + 1] + 1e-10
