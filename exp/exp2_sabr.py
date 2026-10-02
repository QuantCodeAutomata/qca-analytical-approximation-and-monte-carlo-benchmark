"""
Experiment 2: Lognormal SABR model up-and-out put pricing.

Compares Black-Scholes, difference-correction, and gamma-correction
analytical approximations against a controlled Monte Carlo benchmark.
Investigates monitoring-frequency sensitivity (25k vs 12.5k dates).
"""
import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on the path when running from any directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.closed_form import sabr_corrections
from src.control_variates import controlled_price, fit_control_coefficients
from src.metrics import compute_errors, monitoring_sensitivity
from src.sabr_mc import compute_payoffs, simulate_sabr_batch

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NU = 0.2
RHO = -0.9
SIGMA0 = 0.2
S0 = 100.0
B = 120.0
THETA = 1.0

K_GRID = np.arange(80, 122, 2, dtype=float)  # [80, 82, ..., 120] – 21 strikes

V0 = 0.04      # sigma0**2
W0 = 0.2       # sigma0 * sqrt(theta)
B_NORM = 1.0   # (B - S0) / (S0 * W0)

N_PILOT = 10_000
N_PROD = 200_000
N_STEPS = 25_000   # 25 000 monitoring dates (full); 12 500 (half, every 2nd)
SEED_PILOT = 42
SEED_PROD = 123

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 – Analytical approximations
# ---------------------------------------------------------------------------
print("Computing analytical approximations...")
cf = sabr_corrections(S0, K_GRID, B, NU, RHO, SIGMA0, THETA)

# ---------------------------------------------------------------------------
# Step 2 – Pilot Monte Carlo (10 000 paths, seed=42)
# ---------------------------------------------------------------------------
print(f"Running pilot MC ({N_PILOT:,} paths, {N_STEPS:,} steps)...")
S_T_p, S_max_full_p, _ = simulate_sabr_batch(
    n_paths=N_PILOT,
    n_steps=N_STEPS,
    nu=NU,
    rho=RHO,
    sigma0=SIGMA0,
    S0=S0,
    B=B,
    theta=THETA,
    seed=SEED_PILOT,
)
Y_put_p, Y_ctrl_p = compute_payoffs(S_T_p, S_max_full_p, K_GRID, B)

# ---------------------------------------------------------------------------
# Step 3 – Fit control-variate coefficients from pilot paths
# ---------------------------------------------------------------------------
beta_hat = fit_control_coefficients(Y_put_p, Y_ctrl_p)
# At K=B=120, Y_put equals Y_ctrl exactly → beta = 1 by construction.
# The finite-pilot estimator may differ slightly; enforce the exact value.
beta_hat[-1] = 1.0

# ---------------------------------------------------------------------------
# Step 4 – Production Monte Carlo (200 000 paths, seed=123)
# ---------------------------------------------------------------------------
print(f"Running production MC ({N_PROD:,} paths, {N_STEPS:,} steps)...")
S_T, S_max_full, S_max_half = simulate_sabr_batch(
    n_paths=N_PROD,
    n_steps=N_STEPS,
    nu=NU,
    rho=RHO,
    sigma0=SIGMA0,
    S0=S0,
    B=B,
    theta=THETA,
    seed=SEED_PROD,
)

# ---------------------------------------------------------------------------
# Step 5 – Controlled prices: full (25k) and half (12.5k) monitoring
# ---------------------------------------------------------------------------
Y_put_full, Y_ctrl_full = compute_payoffs(S_T, S_max_full, K_GRID, B)
Y_put_half, Y_ctrl_half = compute_payoffs(S_T, S_max_half, K_GRID, B)

price_full, se_full, ci_lo_full, ci_hi_full = controlled_price(
    Y_put_full, Y_ctrl_full, beta_hat, S0, B
)
price_half, se_half, ci_lo_half, ci_hi_half = controlled_price(
    Y_put_half, Y_ctrl_half, beta_hat, S0, B
)

# ---------------------------------------------------------------------------
# Step 6 – Error metrics (vs full-monitoring benchmark)
# ---------------------------------------------------------------------------
prices_analytical = {
    "BS": cf["p_bs"],
    "diff-correction": cf["p_diff"],
    "gamma-correction": cf["p_gamma"],
}
errors_df = compute_errors(prices_analytical, price_full, K_GRID)

# ---------------------------------------------------------------------------
# Step 7 – Monitoring sensitivity
# ---------------------------------------------------------------------------
# True SE of (price_full - price_half): both estimators use the same paths, so
# we compute it from the path-level difference instead of the independence
# upper-bound sqrt(se_full^2 + se_half^2).
Y_diff_paths = Y_put_full - Y_put_half                                # (n, K)
se_diff = Y_diff_paths.std(axis=0, ddof=1) / np.sqrt(N_PROD)

monitor_df = monitoring_sensitivity(
    price_full, price_half, K_GRID,
    se_full=se_diff, se_half=np.zeros_like(se_diff),
)

# ---------------------------------------------------------------------------
# Step 8 – Build prices DataFrame and save CSVs
# ---------------------------------------------------------------------------
prices_df = pd.DataFrame(
    {
        "Strike": K_GRID,
        "MC_Price_25k": price_full,
        "MC_SE_25k": se_full,
        "CI_Lo_25k": ci_lo_full,
        "CI_Hi_25k": ci_hi_full,
        "MC_Price_12k": price_half,
        "MC_SE_12k": se_half,
        "BS": cf["p_bs"],
        "diff_correction": cf["p_diff"],
        "gamma_correction": cf["p_gamma"],
    }
)

prices_df.to_csv(RESULTS_DIR / "exp2_prices.csv", index=False)
errors_df.to_csv(RESULTS_DIR / "exp2_errors.csv", index=False)
monitor_df.to_csv(RESULTS_DIR / "exp2_monitoring.csv", index=False)
print("Saved results/exp2_prices.csv, exp2_errors.csv, exp2_monitoring.csv.")

# ---------------------------------------------------------------------------
# Step 9 – Plot: prices vs strike with 95% CI bands
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(
    K_GRID,
    ci_lo_full,
    ci_hi_full,
    alpha=0.25,
    color="steelblue",
    label="MC 95% CI (25k)",
)
ax.plot(K_GRID, price_full, "o-", color="steelblue", linewidth=2, label="MC (25k monitoring)")
ax.plot(K_GRID, cf["p_bs"],    "s--", color="tab:orange", linewidth=1.5, label="BS")
ax.plot(K_GRID, cf["p_diff"],  "^--", color="tab:green",  linewidth=1.5, label="diff-correction")
ax.plot(K_GRID, cf["p_gamma"], "D--", color="tab:red",    linewidth=1.5, label="gamma-correction")
ax.set_xlabel("Strike $K$")
ax.set_ylabel("Price")
ax.set_title("SABR Up-and-Out Put Prices vs Strike\n"
             r"($\nu=0.2,\;\rho=-0.9,\;\sigma_0=0.2,\;S_0=100,\;B=120,\;\theta=1$)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "exp2_prices.png", dpi=150)
plt.close(fig)
print("Saved results/figures/exp2_prices.png.")

# ---------------------------------------------------------------------------
# Step 10 – Plot: signed errors vs strike
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
ax.plot(K_GRID, cf["p_bs"]    - price_full, "s--", color="tab:orange", linewidth=1.5, label="BS error")
ax.plot(K_GRID, cf["p_diff"]  - price_full, "^--", color="tab:green",  linewidth=1.5, label="diff-correction error")
ax.plot(K_GRID, cf["p_gamma"] - price_full, "D--", color="tab:red",    linewidth=1.5, label="gamma-correction error")
ax.set_xlabel("Strike $K$")
ax.set_ylabel("Signed Error (Analytical $-$ MC)")
ax.set_title("SABR Pricing Errors vs Strike")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "exp2_errors.png", dpi=150)
plt.close(fig)
print("Saved results/figures/exp2_errors.png.")

# ---------------------------------------------------------------------------
# Step 11 – Print summary metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY METRICS")
print("=" * 60)

print("\nAnalytical approximation errors vs MC benchmark (25k monitoring):")
print(f"  {'Method':<20} {'MAE':>10} {'Max AE':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10}")
for method in ["BS", "diff-correction", "gamma-correction"]:
    row = errors_df[errors_df["Method"] == method].iloc[0]
    print(f"  {method:<20} {row['MAE']:>10.5f} {row['MaxAE']:>10.5f}")

print()
print("  Reference values (paper):")
print("    BS            MAE ~0.14937   Max AE ~0.34492")
print("    diff-corr.    MAE ~0.07444   Max AE ~0.09995")
print("    gamma-corr.   MAE ~0.06009   Max AE ~0.09242")

max_diff = monitor_df.attrs.get("max_abs_diff", float("nan"))
max_paired_se = monitor_df.attrs.get("max_paired_se", float("nan"))
print(f"\nMonitoring sensitivity (25k vs 12.5k dates):")
print(f"  Max |price_25k - price_12k| = {max_diff:.5f}   (ref ~0.00408)")
print(f"  Max paired SE               = {max_paired_se:.5f}   (ref ~0.00039)")

print("\nDone.")
