"""
Tests for src/metrics.py.

API:
  compute_errors(prices_analytical: dict[str, ndarray], mc_prices: ndarray,
                 K_grid: ndarray) -> pd.DataFrame
    Columns: Strike, Method, Price, MC_Price, SignedError, AbsError, MAE, MaxAE.

  monitoring_sensitivity(price_full, price_half, K_grid,
                         se_full=None, se_half=None) -> pd.DataFrame
    diff = price_full - price_half; Columns include Diff, AbsDiff, PairedSE.

Mathematical properties tested:
- signed_error = approx - reference
- absolute_error = |signed_error|
- MAE = mean(AbsError) ≤ MaxAE always
- MAE = MaxAE when all errors are equal
- Monitoring diff = price_full - price_half
- PairedSE = sqrt(se_full² + se_half²)
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.metrics import compute_errors, monitoring_sensitivity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
K_GRID = np.arange(80, 122, 2, dtype=float)  # 21 strikes


# ===========================================================================
# Tests for compute_errors
# ===========================================================================

class TestComputeErrors:
    """Tests for compute_errors(prices_analytical, mc_prices, K_grid)."""

    def _make(self, approx: np.ndarray, ref: np.ndarray, name: str = "m") -> pd.DataFrame:
        return compute_errors({name: approx}, ref, K_GRID[: len(approx)])

    def test_returns_dataframe(self):
        approx = np.ones(21) * 2.0
        ref = np.ones(21) * 1.0
        df = compute_errors({"BS": approx}, ref, K_GRID)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = compute_errors({"BS": np.ones(3)}, np.zeros(3), K_GRID[:3])
        for col in ("Strike", "Method", "Price", "MC_Price", "SignedError",
                    "AbsError", "MAE", "MaxAE"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self):
        """With 2 methods and 3 strikes → 6 rows."""
        df = compute_errors(
            {"A": np.ones(3), "B": np.ones(3) * 2.0},
            np.ones(3) * 1.5,
            K_GRID[:3],
        )
        assert len(df) == 6

    def test_signed_error_formula(self):
        """SignedError = Price - MC_Price."""
        approx = np.array([5.0, 3.0, 1.0])
        ref = np.array([4.0, 4.0, 2.0])
        df = compute_errors({"m": approx}, ref, K_GRID[:3])
        expected = approx - ref
        np.testing.assert_allclose(df["SignedError"].values, expected)

    def test_absolute_error_non_negative(self):
        rng = np.random.default_rng(0)
        approx = rng.uniform(0, 5, 21)
        ref = rng.uniform(0, 5, 21)
        df = compute_errors({"m": approx}, ref, K_GRID)
        assert np.all(df["AbsError"].values >= 0.0)

    def test_abs_error_equals_abs_signed(self):
        approx = np.array([1.5, 2.5, 3.5])
        ref = np.array([1.0, 3.0, 4.0])
        df = compute_errors({"m": approx}, ref, K_GRID[:3])
        np.testing.assert_allclose(
            df["AbsError"].values, np.abs(df["SignedError"].values)
        )

    def test_mae_equals_mean_abs_error(self):
        """MAE column should equal mean(AbsError) across all rows of the same method."""
        approx = np.array([1.5, 2.5, 3.5])
        ref = np.array([1.0, 3.0, 4.0])
        df = compute_errors({"m": approx}, ref, K_GRID[:3])
        mae_col = df["MAE"].values
        # All MAE entries for the same method must be identical
        assert np.all(mae_col == mae_col[0])
        # And equal to the mean absolute error
        expected_mae = float(np.mean(np.abs(approx - ref)))
        assert abs(mae_col[0] - expected_mae) < 1e-14

    def test_maxae_equals_max_abs_error(self):
        approx = np.array([1.5, 2.5, 3.5])
        ref = np.array([1.0, 3.0, 4.0])
        df = compute_errors({"m": approx}, ref, K_GRID[:3])
        expected_maxae = float(np.max(np.abs(approx - ref)))
        assert abs(df["MaxAE"].iloc[0] - expected_maxae) < 1e-14

    def test_mae_leq_maxae(self):
        """MAE <= MaxAE always."""
        rng = np.random.default_rng(1)
        approx = rng.uniform(0, 5, 21)
        ref = rng.uniform(0, 5, 21)
        df = compute_errors({"m": approx}, ref, K_GRID)
        assert df["MAE"].iloc[0] <= df["MaxAE"].iloc[0] + 1e-14

    def test_perfect_approximation_zero_errors(self):
        v = np.ones(5) * 3.0
        df = compute_errors({"m": v}, v.copy(), K_GRID[:5])
        np.testing.assert_allclose(df["SignedError"].values, np.zeros(5), atol=1e-14)
        np.testing.assert_allclose(df["AbsError"].values, np.zeros(5), atol=1e-14)
        assert df["MAE"].iloc[0] == pytest.approx(0.0, abs=1e-14)
        assert df["MaxAE"].iloc[0] == pytest.approx(0.0, abs=1e-14)

    def test_constant_error_mae_equals_maxae(self):
        """All errors equal → MAE == MaxAE."""
        approx = np.ones(5) * 2.0
        ref = np.ones(5) * 1.0
        df = compute_errors({"m": approx}, ref, K_GRID[:5])
        assert df["MAE"].iloc[0] == pytest.approx(df["MaxAE"].iloc[0], rel=1e-12)

    def test_strike_column_correct(self):
        """Strike column must match K_GRID."""
        df = compute_errors({"m": np.ones(21)}, np.ones(21), K_GRID)
        np.testing.assert_array_equal(df["Strike"].values, K_GRID)

    def test_multiple_methods_independent_mae(self):
        """Each method gets its own MAE."""
        mc = np.zeros(3)
        df = compute_errors(
            {"bad": np.ones(3) * 5.0, "good": np.ones(3) * 0.1},
            mc, K_GRID[:3],
        )
        mae_bad = df.loc[df["Method"] == "bad", "MAE"].iloc[0]
        mae_good = df.loc[df["Method"] == "good", "MAE"].iloc[0]
        assert mae_bad > mae_good

    def test_regression_known_values(self):
        """Known errors [1, 1, 1, 1, 1] → MAE=1, MaxAE=1."""
        approx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ref = np.array([0.0, 3.0, 2.0, 5.0, 4.0])
        df = compute_errors({"m": approx}, ref, K_GRID[:5])
        assert df["MAE"].iloc[0] == pytest.approx(1.0)
        assert df["MaxAE"].iloc[0] == pytest.approx(1.0)


# ===========================================================================
# Tests for monitoring_sensitivity
# ===========================================================================

class TestMonitoringSensitivity:
    """Tests for monitoring_sensitivity(price_full, price_half, K_grid, ...)."""

    def test_returns_dataframe(self):
        p_full = np.ones(5)
        p_half = np.ones(5) * 1.1
        df = monitoring_sensitivity(p_full, p_half, K_GRID[:5])
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = monitoring_sensitivity(np.ones(3), np.ones(3), K_GRID[:3],
                                    np.ones(3) * 0.01, np.ones(3) * 0.01)
        for col in ("Strike", "Price_25k", "Price_12k", "Diff", "AbsDiff", "PairedSE"):
            assert col in df.columns

    def test_diff_formula(self):
        """Diff = price_full - price_half."""
        p_full = np.array([5.0, 3.0, 1.0])
        p_half = np.array([4.5, 3.2, 1.5])
        df = monitoring_sensitivity(p_full, p_half, K_GRID[:3])
        expected = p_full - p_half
        np.testing.assert_allclose(df["Diff"].values, expected)

    def test_absdiff_non_negative(self):
        rng = np.random.default_rng(0)
        p_full = rng.uniform(0, 5, 21)
        p_half = rng.uniform(0, 5, 21)
        df = monitoring_sensitivity(p_full, p_half, K_GRID)
        assert np.all(df["AbsDiff"].values >= 0.0)

    def test_absdiff_equals_abs_diff(self):
        p_full = np.array([1.0, 2.0, 3.0])
        p_half = np.array([1.2, 1.8, 3.5])
        df = monitoring_sensitivity(p_full, p_half, K_GRID[:3])
        np.testing.assert_allclose(df["AbsDiff"].values, np.abs(df["Diff"].values))

    def test_zero_diff_for_identical_prices(self):
        v = np.arange(1.0, 22.0)
        df = monitoring_sensitivity(v, v.copy(), K_GRID)
        np.testing.assert_allclose(df["Diff"].values, np.zeros(21), atol=1e-14)
        np.testing.assert_allclose(df["AbsDiff"].values, np.zeros(21), atol=1e-14)

    def test_paired_se_formula(self):
        """PairedSE = sqrt(se_full² + se_half²)."""
        p_full = np.ones(3)
        p_half = np.ones(3)
        se_f = np.array([0.02, 0.03, 0.04])
        se_h = np.array([0.04, 0.01, 0.02])
        df = monitoring_sensitivity(p_full, p_half, K_GRID[:3], se_f, se_h)
        expected = np.sqrt(se_f**2 + se_h**2)
        np.testing.assert_allclose(df["PairedSE"].values, expected)

    def test_no_se_columns_when_se_not_provided(self):
        """Without se args, PairedSE column must be absent."""
        df = monitoring_sensitivity(np.ones(5), np.ones(5), K_GRID[:5])
        assert "PairedSE" not in df.columns

    def test_max_abs_diff_attribute(self):
        """DataFrame.attrs['max_abs_diff'] == max(AbsDiff)."""
        p_full = np.array([1.0, 2.0, 3.0])
        p_half = np.array([1.5, 2.2, 3.0])
        df = monitoring_sensitivity(p_full, p_half, K_GRID[:3])
        assert abs(df.attrs["max_abs_diff"] - df["AbsDiff"].max()) < 1e-14

    def test_strike_column_correct(self):
        df = monitoring_sensitivity(np.ones(21), np.ones(21), K_GRID)
        np.testing.assert_array_equal(df["Strike"].values, K_GRID)
