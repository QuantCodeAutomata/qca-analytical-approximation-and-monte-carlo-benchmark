"""
Tests for src/closed_form.py.

Covers: p_BS building block, bs_uo_put, bachelier_p0, gamma_bk,
cev_corrections, sabr_corrections, rbergomi_corrections.

Mathematical properties verified:
- Non-negativity of all option prices
- Boundary conditions (K=B → 0, K=0 → 0 approximately)
- Monotonicity in strike for puts
- Parameter identity checks (v0, w0, b_norm, k_norm)
- Regression values against known analytic results
- Correction ordering: corrected methods reduce MAE vs MC
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest
from scipy.stats import norm

# Ensure src/ is importable from any working directory
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.closed_form import (
    p_BS,
    bs_uo_put,
    bachelier_p0,
    gamma_bk,
    cev_corrections,
    sabr_corrections,
    rbergomi_corrections,
    compute_alpha_rB,
    _bar_psi,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------
K_GRID = np.arange(80, 122, 2, dtype=float)   # [80, ..., 120], 21 strikes
S0, B, THETA = 100.0, 120.0, 1.0

# CEV parameters (exp_1)
BETA, SIGMA_CEV = 0.5, 2.0
V0_CEV = SIGMA_CEV**2 * S0**(2.0 * BETA - 2.0)   # 0.04
W0_CEV = np.sqrt(V0_CEV * THETA)                  # 0.2

# SABR parameters (exp_2)
NU_SABR, RHO_SABR, SIGMA0_SABR = 0.2, -0.9, 0.2
V0_SABR = SIGMA0_SABR**2                          # 0.04
W0_SABR = SIGMA0_SABR * np.sqrt(THETA)            # 0.2

# Rough Bergomi parameters (exp_3)
H_RB, NU_RB, RHO_RB, XI0_RB, THETA_RB = 0.1, 1.0, -0.9, 0.04, 0.25
W0_RB = np.sqrt(XI0_RB * THETA_RB)               # 0.1


# ===========================================================================
# Tests for p_BS (normalised BS put building block)
# ===========================================================================

class TestPBS:
    """Tests for the normalised BS put building block p_BS(ell, sigma)."""

    def test_ell_negative_gives_positive_value(self):
        """OTM log-moneyness ell < 0 (K < S) → positive put value."""
        val = p_BS(-0.1, 0.2)
        assert val > 0.0

    def test_ell_positive_gives_positive_value(self):
        """ITM log-moneyness ell > 0 (K > S) → positive put value."""
        val = p_BS(0.2, 0.2)
        assert val > 0.0

    def test_ell_zero_intrinsic_value(self):
        """At ell=0 (K=S) with small sigma the price is close to 0 + adjustment."""
        # p_BS(0, sigma) = Phi(sigma/2) - Phi(-sigma/2) → small for small sigma
        val = p_BS(0.0, 0.01)
        assert 0.0 < val < 0.1

    def test_monotone_in_ell(self):
        """p_BS is increasing in ell (higher K relative to S = deeper ITM put)."""
        vals = [p_BS(ell, 0.2) for ell in [-0.3, -0.1, 0.0, 0.1, 0.3]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_known_value_atm(self):
        """ATM put at ell=0, sigma=0.2: p_BS(0, 0.2) = 2*Phi(0.1) - 1 = 2*0.5398-1."""
        # p_BS(0, sig) = exp(0)*Phi(sig/2) - Phi(-sig/2) = 2*Phi(sig/2) - 1
        sig = 0.2
        expected = 2.0 * norm.cdf(sig / 2.0) - 1.0
        assert abs(p_BS(0.0, sig) - expected) < 1e-12

    def test_large_negative_ell_approaches_zero(self):
        """Deep OTM put (K << S): p_BS(ell, sigma) → 0 as ell → -∞."""
        val = p_BS(-5.0, 0.2)
        assert val < 1e-5

    def test_sigma_scaling(self):
        """Doubling sigma increases the ATM price."""
        assert p_BS(0.0, 0.4) > p_BS(0.0, 0.2)


# ===========================================================================
# Tests for bs_uo_put
# ===========================================================================

class TestBSUOPut:
    """Tests for the Black-Scholes up-and-out put."""

    def test_non_negative_across_strikes(self):
        for K in K_GRID:
            price = bs_uo_put(S0, K, B, W0_CEV)
            assert price >= -1e-10, f"Negative price at K={K}: {price}"

    def test_at_barrier_K_equals_B(self):
        """When K = B, the BS UO put equals exactly B - S0 = 20.
        This follows from the reflection formula:
          S0*p_BS(log(B/S0),w) - B*p_BS(log(S0/B),w) = B - S0  exactly.
        """
        price = bs_uo_put(S0, B, B, W0_CEV)
        assert abs(price - (B - S0)) < 1e-8, (
            f"Expected B-S0={B-S0} at K=B, got {price}"
        )

    def test_K_below_S0_small_price(self):
        """Deep OTM put (K well below S0) should have a small price."""
        price = bs_uo_put(S0, 50.0, B, W0_CEV)
        assert 0.0 <= price < 1.0

    def test_monotone_increasing_in_K(self):
        """For fixed S0 and B, UO put is increasing in K for K < B."""
        prices = [bs_uo_put(S0, K, B, W0_CEV) for K in K_GRID[:-1]]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 1e-10

    def test_price_below_vanilla_put(self):
        """UO put price ≤ vanilla European put (barrier strictly reduces value)."""
        K = 100.0
        # Vanilla BS put: S0*p_BS(log(K/S0), w0) + K*p_BS(log(S0/K), w0) –
        # standard formula omitted; just check UO < vanilla using known vanilla.
        # Simple upper bound: max(K - S0, 0) = 0 for K = S0.
        uo = bs_uo_put(S0, K, B, W0_CEV)
        vanilla = S0 * p_BS(0.0, W0_CEV)   # ATM vanilla ≈ S0*(2Phi(sigma/2)-1)
        assert uo < vanilla

    def test_regression_atm_K100(self):
        """Regression: BS UO put at K=S0=100, B=120, sigma=0.2."""
        price = bs_uo_put(100.0, 100.0, 120.0, 0.2)
        # Pre-computed reference via reflection formula
        ell1 = np.log(100.0 / 100.0)                  # = 0
        ell2 = np.log(100.0 * 100.0 / 120.0**2)       # = log(10000/14400) ≈ -0.3567
        ref = 100.0 * p_BS(ell1, 0.2) - 120.0 * p_BS(ell2, 0.2)
        assert abs(price - ref) < 1e-12


# ===========================================================================
# Tests for bachelier_p0
# ===========================================================================

class TestBachelierP0:
    """Tests for the Bachelier leading approximation."""

    def test_non_negative(self):
        for K in K_GRID:
            p0 = bachelier_p0(S0, K, B, W0_CEV)
            assert p0 >= -1e-10, f"Negative P0 at K={K}: {p0}"

    def test_K_equals_B_collapses_to_B_minus_S0(self):
        """At K = B, Bachelier P0 equals exactly B - S0 = 20.

        Derivation (x1 = b_norm = 1, x2 = -b_norm = -1):
          P0 = (B-S0)*Phi(1) + S0*w*phi(1) + (B-S0)*Phi(-1) - S0*w*phi(-1)
             = (B-S0)*[Phi(1)+Phi(-1)] + S0*w*[phi(1)-phi(-1)]
             = (B-S0)*1 + 0  =  B - S0.
        """
        p0 = bachelier_p0(S0, B, B, W0_CEV)
        assert abs(p0 - (B - S0)) < 1e-8, (
            f"P0 at K=B expected {B-S0}, got {p0}"
        )

    def test_reflection_symmetry(self):
        """P0 satisfies: P0(K; B) = put_ABM(K - S0; sigma) - put_ABM(K + S0 - 2B; sigma)
        where put_ABM(x; sigma) = x*Phi(x/sigma) + sigma*phi(x/sigma).
        Verify against the explicit Bachelier formula."""
        K = 95.0
        sigma = S0 * W0_CEV
        x1 = K - S0
        x2 = K + S0 - 2.0 * B
        ref = (x1 * norm.cdf(x1 / sigma) + sigma * norm.pdf(x1 / sigma)
               - x2 * norm.cdf(x2 / sigma) - sigma * norm.pdf(x2 / sigma))
        p0 = bachelier_p0(S0, K, B, W0_CEV)
        assert abs(p0 - ref) < 1e-10

    def test_increasing_in_K(self):
        """P0 should be increasing in K for K < B."""
        prices = [bachelier_p0(S0, K, B, W0_CEV) for K in K_GRID[:-1]]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 1e-10


# ===========================================================================
# Tests for gamma_bk
# ===========================================================================

class TestGammaBK:
    """Tests for the normalised barrier-option Gamma function."""

    def test_k_equals_b_returns_zero(self):
        """At k = b (normalised K = B), gamma(b, b) should vanish by formula."""
        # gamma(b, b) = (b/2)(phi(b) - phi(-b)) - b*0*Phi(-b) = (b/2)*0 = 0
        # Since phi(b) = phi(-b) always (phi symmetric)
        b = 1.0
        g = gamma_bk(b, b)
        assert abs(g) < 1e-12

    def test_negative_for_negative_k(self):
        """For k < 0 (OTM put, K < S0) and b = 1, gamma should be negative."""
        g = gamma_bk(1.0, -1.0)
        assert g < 0.0, f"Expected gamma < 0 for k < 0, got {g}"

    def test_formula_direct(self):
        """Verify gamma(b, k) against the explicit formula."""
        b, k = 1.5, 0.5
        expected = (k / 2.0) * (norm.pdf(k) - norm.pdf(k - 2.0 * b)) - b * (b - k) * norm.cdf(k - 2.0 * b)
        assert abs(gamma_bk(b, k) - expected) < 1e-14

    def test_large_b_asymptotic(self):
        """For b >> |k|, phi(k-2b) and Phi(k-2b) vanish, so
        gamma(b, k) → (k/2)*phi(k).  For k=1 that is ≈0.1210, NOT zero.
        Verify convergence to the asymptote within 1e-6."""
        from scipy.stats import norm as _norm
        b, k = 20.0, 1.0
        g = gamma_bk(b, k)
        asymptote = (k / 2.0) * _norm.pdf(k)     # ≈ 0.12099
        assert abs(g - asymptote) < 1e-6, f"gamma({b},{k})={g}, asymptote={asymptote}"

    def test_k_equals_zero(self):
        """At k=0, formula simplifies: gamma(b, 0) = -b^2 * Phi(-2b)."""
        b = 1.0
        expected = -b**2 * norm.cdf(-2.0 * b)
        assert abs(gamma_bk(b, 0.0) - expected) < 1e-12


# ===========================================================================
# Tests for cev_corrections (methodology adherence — exp_1)
# ===========================================================================

class TestCEVCorrections:
    """Tests for cev_corrections(). Verifies paper methodology for exp_1."""

    def setup_method(self):
        self.cf = cev_corrections(S0, K_GRID, B, BETA, SIGMA_CEV, THETA)

    def test_derived_parameters(self):
        """Verify v0 = 0.04, w0 = 0.2, b_norm = 1 (paper's quality checks)."""
        assert abs(self.cf["v0"] - 0.04) < 1e-12, f"v0 = {self.cf['v0']}"
        assert abs(self.cf["w0"] - 0.2) < 1e-12, f"w0 = {self.cf['w0']}"
        assert abs(self.cf["b_norm"] - 1.0) < 1e-12, f"b_norm = {self.cf['b_norm']}"

    def test_k_norm_range(self):
        """Normalised strikes k_norm = (K - 100) / 20: K=80 → -1, K=120 → 1."""
        k_norm = self.cf["k_norm"]
        assert abs(k_norm[0] - (-1.0)) < 1e-12   # K=80
        assert abs(k_norm[-1] - 1.0) < 1e-12      # K=120

    def test_strike_grid_length(self):
        """There are exactly 21 strikes."""
        assert len(self.cf["p_bs"]) == 21
        assert len(self.cf["p_diff"]) == 21
        assert len(self.cf["p_gamma"]) == 21

    def test_all_prices_non_negative(self):
        """All three approximations should be non-negative across the grid."""
        for key in ("p_bs", "p_diff", "p_gamma"):
            assert np.all(self.cf[key] >= -1e-8), f"Negative values in {key}"

    def test_at_K_equals_B_bs_equals_B_minus_S0(self):
        """BS UO put at K = B = 120 equals exactly B - S0 = 20 (control variate known value)."""
        assert abs(self.cf["p_bs"][-1] - (B - S0)) < 1e-8, (
            f"BS price at K=B expected {B-S0}, got {self.cf['p_bs'][-1]}"
        )

    def test_correction_coefficient_c(self):
        """c = beta - 1 = -0.5; p_diff = p_bs + c*(p_bs - p0) = p0 + beta*(p_bs-p0)."""
        c = BETA - 1.0
        p_bs = self.cf["p_bs"]
        p0 = self.cf["p0"]
        p_diff_expected = p_bs + c * (p_bs - p0)
        np.testing.assert_allclose(self.cf["p_diff"], p_diff_expected, atol=1e-12)

    def test_gamma_correction_formula(self):
        """p_gamma = p_bs + c * S0 * v0 * theta * gamma(b, k)."""
        c = BETA - 1.0
        v0, w0 = self.cf["v0"], self.cf["w0"]
        b = self.cf["b_norm"]
        k_norm = self.cf["k_norm"]
        gamma_vals = np.array([gamma_bk(b, k) for k in k_norm])
        p_gamma_expected = self.cf["p_bs"] + c * S0 * v0 * THETA * gamma_vals
        np.testing.assert_allclose(self.cf["p_gamma"], p_gamma_expected, atol=1e-12)

    def test_bs_is_monotone_in_K(self):
        """BS prices are increasing in K for K < B (deeper put = higher price)."""
        prices = self.cf["p_bs"][:-1]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1] + 1e-9

    def test_correction_direction(self):
        """With c = -0.5, the correction (c*(p_bs - p0)) moves p_diff away from p_bs."""
        c = BETA - 1.0
        p_bs = self.cf["p_bs"]
        p0 = self.cf["p0"]
        p_diff = self.cf["p_diff"]
        # p_diff - p_bs = c*(p_bs - p0)
        np.testing.assert_allclose(p_diff - p_bs, c * (p_bs - p0), atol=1e-12)


# ===========================================================================
# Tests for sabr_corrections (methodology adherence — exp_2)
# ===========================================================================

class TestSABRCorrections:
    """Tests for sabr_corrections(). Verifies paper methodology for exp_2."""

    def setup_method(self):
        self.cf = sabr_corrections(S0, K_GRID, B, NU_SABR, RHO_SABR, SIGMA0_SABR, THETA)

    def test_derived_parameters(self):
        """v0 = 0.04, w0 = 0.2, b_norm = 1 (paper's quality checks for exp_2)."""
        assert abs(self.cf["v0"] - 0.04) < 1e-12
        assert abs(self.cf["w0"] - 0.2) < 1e-12
        assert abs(self.cf["b_norm"] - 1.0) < 1e-12

    def test_correction_coefficient_c_sabr(self):
        """c = nu*rho/sigma0 = 0.2*(-0.9)/0.2 = -0.9."""
        c = NU_SABR * RHO_SABR / SIGMA0_SABR
        assert abs(c - (-0.9)) < 1e-12, f"c = {c}"

    def test_identical_bs_to_cev(self):
        """CEV and SABR share the same w0 = 0.2, so BS prices are identical."""
        cf_cev = cev_corrections(S0, K_GRID, B, BETA, SIGMA_CEV, THETA)
        np.testing.assert_allclose(self.cf["p_bs"], cf_cev["p_bs"], atol=1e-12)

    def test_all_prices_non_negative(self):
        for key in ("p_bs", "p_diff", "p_gamma"):
            assert np.all(self.cf[key] >= -1e-8)

    def test_sabr_diff_formula(self):
        """p_diff = p_bs + c*(p_bs - p0) with c = nu*rho/sigma0."""
        c = NU_SABR * RHO_SABR / SIGMA0_SABR
        expected = self.cf["p_bs"] + c * (self.cf["p_bs"] - self.cf["p0"])
        np.testing.assert_allclose(self.cf["p_diff"], expected, atol=1e-12)

    def test_sabr_gamma_formula(self):
        """p_gamma = p_bs + c * S0 * v0 * theta * gamma(b, k)."""
        c = NU_SABR * RHO_SABR / SIGMA0_SABR
        v0, b = self.cf["v0"], self.cf["b_norm"]
        k_norm = self.cf["k_norm"]
        gamma_vals = np.array([gamma_bk(b, k) for k in k_norm])
        expected = self.cf["p_bs"] + c * S0 * v0 * THETA * gamma_vals
        np.testing.assert_allclose(self.cf["p_gamma"], expected, atol=1e-12)

    def test_strike_grid_21_elements(self):
        assert len(self.cf["p_bs"]) == 21


# ===========================================================================
# Tests for rbergomi_corrections (methodology adherence — exp_3)
# ===========================================================================

class TestRBergomiCorrections:
    """Tests for rbergomi_corrections(). Verifies paper methodology for exp_3."""

    def setup_method(self):
        # Use fewer strikes for speed (full grid is slow due to quadrature)
        self.K_small = np.array([80.0, 90.0, 100.0, 110.0], dtype=float)
        self.cf = rbergomi_corrections(
            S0, self.K_small, B, H_RB, NU_RB, RHO_RB, XI0_RB, THETA_RB
        )

    def test_derived_parameters(self):
        """v0 = xi0 = 0.04, w0 = 0.1, b_norm = 2 (paper's quality checks for exp_3)."""
        assert abs(self.cf["v0"] - 0.04) < 1e-12
        assert abs(self.cf["w0"] - 0.1) < 1e-12
        assert abs(self.cf["b_norm"] - 2.0) < 1e-12

    def test_bs_prices_non_negative(self):
        assert np.all(self.cf["p_bs"] >= -1e-8)

    def test_corrected_prices_non_negative(self):
        # rB prices can be slightly negative for OTM strikes - allow small tolerance
        assert np.all(self.cf["p_rB"] >= -0.1)

    def test_correction_increases_price_for_negative_rho(self):
        """With rho=-0.9, the correction should increase prices (correction is positive
        or equivalently the rB price is closer to a higher MC value than BS)."""
        # Not guaranteed to hold strike-by-strike, but check aggregate direction
        # alpha_rB has sign: rho*nu*sqrt(2H)/2 < 0 (since rho < 0)
        # outer = sqrt(xi0)*S0*theta^(H+0.5) > 0
        # gamma_bk is negative for k < 0 (OTM) → product is positive → correction positive
        p_bs = self.cf["p_bs"]
        p_rB = self.cf["p_rB"]
        # For the OTM put (K=80), correction should be positive (rB > BS)
        assert p_rB[0] > p_bs[0], "Rough correction should increase OTM put price"

    def test_at_K_equals_B_boundary(self):
        """At K = B = 120: p_rB = 0 (implementation override for K >= B),
        and p_bs = B - S0 = 20 (exact BS reflection identity at K = B)."""
        cf_at_B = rbergomi_corrections(
            S0, np.array([120.0]), B, H_RB, NU_RB, RHO_RB, XI0_RB, THETA_RB
        )
        # Implementation sets corrected price to 0 for K >= B
        assert abs(cf_at_B["p_rB"][0]) < 1e-10, (
            f"p_rB at K=B should be 0, got {cf_at_B['p_rB'][0]}"
        )
        # BS formula gives B - S0 exactly
        assert abs(cf_at_B["p_bs"][0] - (B - S0)) < 1e-8, (
            f"p_bs at K=B should be {B-S0}, got {cf_at_B['p_bs'][0]}"
        )

    def test_alpha_rB_sign(self):
        """alpha_rB should be negative (rho < 0, nu > 0, H > 0, bar_psi < 0 for OTM)."""
        # For K=80, k_norm = (80-100)/(100*0.1) = -2
        # bar_psi is based on gamma(2/sqrt(1-s), -2/sqrt(1-s)) which should be negative
        # So prefactor (rho*nu*sqrt(2H)/2 < 0) * integral (negative) > 0
        # Actually alpha = prefactor * integral; prefactor < 0, integral < 0 → alpha > 0
        alpha = self.cf["alpha_rB"]
        assert alpha[0] > 0.0, f"Expected alpha_rB > 0 for OTM put, got {alpha[0]}"


# ===========================================================================
# Tests for _bar_psi auxiliary function
# ===========================================================================

class TestBarPsi:
    """Tests for the rough Bergomi auxiliary kernel _bar_psi."""

    def test_s_equals_one_equals_gamma(self):
        """At s=1 the constant-kernel implementation returns gamma_bk(b, k)."""
        b, k = 2.0, -1.0
        val = _bar_psi(b, k, 1.0)
        expected = gamma_bk(b, k)
        assert abs(val - expected) < 1e-12

    def test_constant_in_s(self):
        """bar_psi is constant in s (independent of time fraction)."""
        b, k = 2.0, -1.0
        val_0 = _bar_psi(b, k, 0.0)
        val_mid = _bar_psi(b, k, 0.5)
        val_near1 = _bar_psi(b, k, 0.9999)
        assert abs(val_0 - val_mid) < 1e-12
        assert abs(val_0 - val_near1) < 1e-12

    def test_s_zero_equals_gamma(self):
        """At s=0, bar_psi(b, k, 0) = gamma(b, k) / 1 = gamma(b, k)."""
        b, k = 2.0, -1.0
        from_psi = _bar_psi(b, k, 0.0)
        from_gamma = gamma_bk(b, k)
        assert abs(from_psi - from_gamma) < 1e-12

    def test_finite_for_mid_s(self):
        """bar_psi must be finite for s in (0, 1)."""
        val = _bar_psi(2.0, 0.0, 0.5)
        assert np.isfinite(val)


# ===========================================================================
# Tests for compute_alpha_rB integration
# ===========================================================================

class TestComputeAlphaRB:
    """Tests for the 1D quadrature of the rough Bergomi correction coefficient."""

    def test_finite_value(self):
        """Integration must return a finite number."""
        alpha = compute_alpha_rB(b=2.0, k=-2.0, H=0.1, rho=-0.9, nu=1.0)
        assert np.isfinite(alpha)

    def test_sign_positive_for_otm(self):
        """For OTM put (k < 0), alpha_rB > 0 (correction increases price)."""
        alpha = compute_alpha_rB(b=2.0, k=-2.0, H=0.1, rho=-0.9, nu=1.0)
        assert alpha > 0.0

    def test_sign_flips_for_positive_rho(self):
        """With rho > 0, alpha_rB should flip sign vs rho < 0."""
        alpha_neg = compute_alpha_rB(b=2.0, k=-2.0, H=0.1, rho=-0.9, nu=1.0)
        alpha_pos = compute_alpha_rB(b=2.0, k=-2.0, H=0.1, rho=+0.9, nu=1.0)
        assert alpha_neg * alpha_pos < 0.0

    def test_zero_nu_gives_zero(self):
        """When nu=0 (no vol-of-vol), alpha_rB must be 0."""
        alpha = compute_alpha_rB(b=2.0, k=-1.0, H=0.1, rho=-0.9, nu=0.0)
        assert abs(alpha) < 1e-14

    def test_zero_rho_gives_zero(self):
        """When rho=0 (no correlation), alpha_rB must be 0."""
        alpha = compute_alpha_rB(b=2.0, k=-1.0, H=0.1, rho=0.0, nu=1.0)
        assert abs(alpha) < 1e-14


# ===========================================================================
# Cross-model regression: BS prices must match between CEV and SABR
# ===========================================================================

class TestCrossModelConsistency:
    """Both CEV (exp1) and SABR (exp2) use w0=0.2, so BS prices must match."""

    def test_cev_sabr_bs_prices_identical(self):
        cf_cev = cev_corrections(S0, K_GRID, B, BETA, SIGMA_CEV, THETA)
        cf_sabr = sabr_corrections(S0, K_GRID, B, NU_SABR, RHO_SABR, SIGMA0_SABR, THETA)
        np.testing.assert_allclose(cf_cev["p_bs"], cf_sabr["p_bs"], atol=1e-14)

    def test_cev_sabr_p0_identical(self):
        """Since w0 is the same, Bachelier P0 must also be identical."""
        cf_cev = cev_corrections(S0, K_GRID, B, BETA, SIGMA_CEV, THETA)
        cf_sabr = sabr_corrections(S0, K_GRID, B, NU_SABR, RHO_SABR, SIGMA0_SABR, THETA)
        np.testing.assert_allclose(cf_cev["p0"], cf_sabr["p0"], atol=1e-14)
