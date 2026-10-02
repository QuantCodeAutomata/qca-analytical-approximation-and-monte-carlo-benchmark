"""
Tests for src/control_variates.py.

Covers: fit_control_coefficients, controlled_price.

Key properties:
- Variance-minimising beta_hat = Cov(Y_K, Y_B) / Var(Y_B)
- At K = B, Y_put == Y_ctrl, so beta_hat = 1 and the controlled price collapses
  exactly to B - S0 with SE = 0 (after we override beta_hat[-1] = 1)
- For uncorrelated Y_put and Y_ctrl, beta_hat = 0 and price = mean(Y_put)
- Standard errors are positive and confidence intervals are correctly centred
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.control_variates import fit_control_coefficients, controlled_price


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(0)
N = 50_000
K_GRID = np.arange(80, 122, 2, dtype=float)
S0, B = 100.0, 120.0


def _make_payoffs(n: int, n_strikes: int, seed: int = 7):
    """Generate synthetic payoffs: Y_put ~ Uniform[0,5], Y_ctrl ~ Uniform[0,5]
    with a known correlation structure."""
    rng = np.random.default_rng(seed)
    Y_ctrl = rng.uniform(0.0, 5.0, size=n)
    # Y_put is correlated with Y_ctrl at each strike
    Y_put = np.zeros((n, n_strikes))
    for i in range(n_strikes):
        noise = rng.standard_normal(n)
        Y_put[:, i] = 0.6 * Y_ctrl + 0.8 * noise + float(i) * 0.01
        Y_put[:, i] = np.maximum(Y_put[:, i], 0.0)
    return Y_put, Y_ctrl


# ===========================================================================
# Tests for fit_control_coefficients
# ===========================================================================

class TestFitControlCoefficients:

    def test_shape(self):
        """Result has shape (n_strikes,)."""
        Y_put, Y_ctrl = _make_payoffs(1000, 5)
        beta = fit_control_coefficients(Y_put, Y_ctrl)
        assert beta.shape == (5,)

    def test_zero_variance_ctrl_gives_zero(self):
        """If Y_ctrl is constant, Var = 0 and beta_hat should return zeros."""
        Y_put = np.ones((100, 3))
        Y_ctrl = np.full(100, 5.0)
        beta = fit_control_coefficients(Y_put, Y_ctrl)
        np.testing.assert_array_equal(beta, np.zeros(3))

    def test_beta_hat_equals_1_when_Y_put_equals_Y_ctrl(self):
        """At K = B, Y_put = Y_ctrl exactly, so beta_hat = Cov(Y,Y)/Var(Y) = 1."""
        rng = np.random.default_rng(42)
        Y_ctrl = rng.uniform(0.0, 10.0, size=10_000)
        # Single-strike case: Y_put[:, 0] = Y_ctrl exactly
        Y_put = Y_ctrl[:, None].copy()
        beta = fit_control_coefficients(Y_put, Y_ctrl)
        assert abs(beta[0] - 1.0) < 0.01   # Should be very close to 1

    def test_beta_hat_zero_for_uncorrelated(self):
        """Uncorrelated payoffs yield beta_hat ≈ 0."""
        rng = np.random.default_rng(99)
        n = 100_000
        Y_ctrl = rng.standard_normal(n)
        Y_put = rng.standard_normal((n, 1))   # independent of Y_ctrl
        beta = fit_control_coefficients(Y_put, Y_ctrl)
        assert abs(beta[0]) < 0.05   # Should be near 0 for large n

    def test_positive_beta_for_positively_correlated(self):
        """Positively correlated Y_put and Y_ctrl → positive beta_hat."""
        rng = np.random.default_rng(11)
        n = 5000
        Y_ctrl = rng.standard_normal(n)
        Y_put = (Y_ctrl + 0.1 * rng.standard_normal(n))[:, None]
        beta = fit_control_coefficients(Y_put, Y_ctrl)
        assert beta[0] > 0.0

    def test_exact_formula_small_case(self):
        """Verify beta_hat against Cov(Y_K, Y_B)/Var(Y_B) formula directly."""
        rng = np.random.default_rng(22)
        n = 500
        Y_ctrl = rng.standard_normal(n)
        Y_put = np.column_stack([2.0 * Y_ctrl + rng.standard_normal(n)])
        beta = fit_control_coefficients(Y_put, Y_ctrl)

        # Manual calculation
        cov = np.mean((Y_put[:, 0] - Y_put[:, 0].mean()) * (Y_ctrl - Y_ctrl.mean()))
        var = np.var(Y_ctrl, ddof=1)
        expected = cov / var
        assert abs(beta[0] - expected) < 1e-12

    def test_input_validation_ndim(self):
        """Y_put must be 2D; Y_ctrl must be 1D."""
        Y_put = np.ones((100, 3))
        Y_ctrl_2d = np.ones((100, 1))
        with pytest.raises(AssertionError):
            fit_control_coefficients(Y_put, Y_ctrl_2d)

    def test_input_validation_shape_mismatch(self):
        """Mismatched first dimensions should raise AssertionError."""
        Y_put = np.ones((100, 3))
        Y_ctrl = np.ones(200)
        with pytest.raises(AssertionError):
            fit_control_coefficients(Y_put, Y_ctrl)


# ===========================================================================
# Tests for controlled_price
# ===========================================================================

class TestControlledPrice:

    def test_output_shapes(self):
        """All outputs have shape (n_strikes,)."""
        n, n_k = 500, 5
        Y_put = np.random.default_rng(3).uniform(0, 5, (n, n_k))
        Y_ctrl = np.random.default_rng(4).uniform(0, 10, n)
        beta_hat = np.zeros(n_k)
        price, se, ci_lo, ci_hi = controlled_price(Y_put, Y_ctrl, beta_hat, S0, B)
        for arr in (price, se, ci_lo, ci_hi):
            assert arr.shape == (n_k,), f"Wrong shape: {arr.shape}"

    def test_zero_beta_hat_returns_sample_mean(self):
        """With beta_hat = 0, controlled price = mean(Y_put) at each strike."""
        rng = np.random.default_rng(5)
        n, n_k = 1000, 3
        Y_put = rng.uniform(0, 5, (n, n_k))
        Y_ctrl = rng.uniform(0, 10, n)
        beta_hat = np.zeros(n_k)
        price, se, ci_lo, ci_hi = controlled_price(Y_put, Y_ctrl, beta_hat, S0, B)
        np.testing.assert_allclose(price, Y_put.mean(axis=0), atol=1e-12)

    def test_se_positive(self):
        """Standard errors must be positive."""
        rng = np.random.default_rng(6)
        n, n_k = 2000, 5
        Y_put = rng.uniform(0, 5, (n, n_k))
        Y_ctrl = rng.uniform(0, 10, n)
        beta_hat = np.ones(n_k) * 0.3
        price, se, ci_lo, ci_hi = controlled_price(Y_put, Y_ctrl, beta_hat, S0, B)
        assert np.all(se > 0.0)

    def test_ci_contains_price(self):
        """ci_lo <= price <= ci_hi at every strike."""
        rng = np.random.default_rng(7)
        n, n_k = 1000, 4
        Y_put = rng.uniform(0, 5, (n, n_k))
        Y_ctrl = rng.uniform(0, 10, n)
        beta_hat = np.ones(n_k) * 0.5
        price, se, ci_lo, ci_hi = controlled_price(Y_put, Y_ctrl, beta_hat, S0, B)
        assert np.all(ci_lo <= price + 1e-12)
        assert np.all(ci_hi >= price - 1e-12)

    def test_ci_width_equals_1_96_times_se(self):
        """CI width should be 2 * 1.96 * SE."""
        rng = np.random.default_rng(8)
        n, n_k = 1000, 4
        Y_put = rng.uniform(0, 5, (n, n_k))
        Y_ctrl = rng.uniform(0, 10, n)
        beta_hat = np.ones(n_k) * 0.5
        price, se, ci_lo, ci_hi = controlled_price(Y_put, Y_ctrl, beta_hat, S0, B)
        np.testing.assert_allclose(ci_hi - ci_lo, 2.0 * 1.96 * se, rtol=1e-10)

    def test_at_K_equals_B_collapses(self):
        """When Y_put[:, -1] = Y_ctrl exactly and beta_hat[-1] = 1,
        controlled price = B - S0 exactly."""
        rng = np.random.default_rng(9)
        n = 50_000
        Y_ctrl = rng.uniform(0.0, 10.0, n)
        n_k = 3
        Y_put = np.zeros((n, n_k))
        Y_put[:, 0] = rng.uniform(0.0, 5.0, n)
        Y_put[:, 1] = rng.uniform(0.0, 8.0, n)
        Y_put[:, 2] = Y_ctrl.copy()   # Last strike mimics K=B

        beta_hat = np.array([0.3, 0.6, 1.0])  # Last is enforced 1.0

        price, se, ci_lo, ci_hi = controlled_price(Y_put, Y_ctrl, beta_hat, S0, B)
        # At last strike: price must be exactly B - S0 = 20.0
        assert abs(price[-1] - (B - S0)) < 1e-10, f"Expected {B-S0}, got {price[-1]}"
        # SE at last strike must be 0
        assert se[-1] < 1e-12, f"Expected SE=0, got {se[-1]}"

    def test_variance_reduction_effect(self):
        """The controlled estimator should have lower variance than the raw mean
        when Y_put and Y_ctrl are highly correlated."""
        rng = np.random.default_rng(10)
        n = 100_000
        # Y_ctrl and Y_put[i, 0] are almost perfectly correlated
        base = rng.standard_normal(n)
        Y_ctrl = base + 0.01 * rng.standard_normal(n)
        Y_put = (base + 0.01 * rng.standard_normal(n))[:, None] + 2.0

        beta_hat = np.array([1.0])
        price_cv, se_cv, _, _ = controlled_price(Y_put, Y_ctrl, beta_hat, S0, B)

        # Without CV (beta=0): SE = std(Y_put) / sqrt(n)
        se_raw = Y_put[:, 0].std(ddof=1) / np.sqrt(n)

        # With near-perfect correlation, CV SE should be << raw SE
        assert se_cv[0] < se_raw * 0.1, (
            f"CV SE {se_cv[0]:.6f} not much smaller than raw SE {se_raw:.6f}"
        )
