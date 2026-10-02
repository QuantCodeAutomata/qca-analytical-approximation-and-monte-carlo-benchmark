"""
Experiment 3: Rough Bergomi model – up-and-out put pricing.

Steps:
  1. Compute analytical Black-Scholes baseline and rough Bergomi correction.
  2. Run pilot MC (10 000 paths) to fit control-variate coefficients.
  3. Run production MC (200 000 paths) under full (25k) and half (12.5k) monitoring.
  4. Compute controlled prices, error metrics, and monitoring-sensitivity table.
  5. Save results/exp3_prices.csv, results/exp3_errors.csv, results/exp3_monitoring.csv.
  6. Save results/figures/exp3_prices.png and results/figures/exp3_errors.png.
  7. Print summary metrics alongside expected reference values.
"""
from __future__ import annotations

import os
import sys

# Resolve project root (parent of exp/) so src.* imports work from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rbergomi_mc import simulate_rbergomi_batch, compute_payoffs
from src.closed_form import rbergomi_corrections
from src.control_variates import fit_control_coefficients, controlled_price
from src.metrics import compute_errors, monitoring_sensitivity

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
H = 0.1
nu = 1.0
rho = -0.9
xi0 = 0.04
S0 = 100.0
B = 120.0
theta = 0.25

K_grid = np.arange(80, 122, 2, dtype=float)   # [80, 82, ..., 120] – 21 strikes
n_strikes = len(K_grid)

# Derived convenience constants (checked against model params)
v0 = 0.04          # = xi0
w0 = 0.1           # = sqrt(v0 * theta) = sqrt(0.04 * 0.25)
b_norm = 2.0       # = (B - S0) / (S0 * w0) = 20 / 10

N_STEPS = 25_000
N_PILOT = 10_000
N_PROD = 200_000
BATCH = 1_000

print("=" * 60)
print("Experiment 3 – Rough Bergomi up-and-out put pricing")
print("=" * 60)
print(f"  H={H}, nu={nu}, rho={rho}, xi0={xi0}")
print(f"  S0={S0}, B={B}, theta={theta}")
print(f"  K_grid: {K_grid[0]:.0f} to {K_grid[-1]:.0f}  ({n_strikes} strikes)")
print(f"  n_steps={N_STEPS}, N_pilot={N_PILOT}, N_prod={N_PROD}, batch={BATCH}")

# ---------------------------------------------------------------------------
# Step 1 – Analytical corrections
# ---------------------------------------------------------------------------
print("\n[1] Computing analytical corrections (BS + rough Bergomi)…")
cf = rbergomi_corrections(S0, K_grid, B, H, nu, rho, xi0, theta)
p_bs = cf["p_bs"]      # shape (n_strikes,)
p_rB = cf["p_rB"]      # shape (n_strikes,)
print(f"    Done. p_bs range: [{p_bs.min():.4f}, {p_bs.max():.4f}]")
print(f"    Done. p_rB range: [{p_rB.min():.4f}, {p_rB.max():.4f}]")

# ---------------------------------------------------------------------------
# Step 2 – Pilot MC: fit control coefficients (full monitoring)
# ---------------------------------------------------------------------------
print(f"\n[2] Pilot MC ({N_PILOT} paths, seed=42, batch={BATCH})…")
S_T_p, S_max_full_p, S_max_half_p = simulate_rbergomi_batch(
    n_paths=N_PILOT,
    n_steps=N_STEPS,
    H=H,
    nu=nu,
    rho=rho,
    xi0=xi0,
    S0=S0,
    B=B,
    theta=theta,
    seed=42,
    batch_size=BATCH,
)
Y_put_p, Y_ctrl_p = compute_payoffs(S_T_p, S_max_full_p, K_grid, B)
beta_hat = fit_control_coefficients(Y_put_p, Y_ctrl_p)
# At K=B=120, Y_put equals Y_ctrl exactly → beta = 1 by construction.
# Enforce the exact value to avoid finite-pilot rounding errors.
beta_hat[-1] = 1.0
print(f"    beta_hat range: [{beta_hat.min():.4f}, {beta_hat.max():.4f}]")

# ---------------------------------------------------------------------------
# Step 3 – Production MC (full + half monitoring)
# ---------------------------------------------------------------------------
print(f"\n[3] Production MC ({N_PROD} paths, seed=123, batch={BATCH})…")
S_T, S_max_full, S_max_half = simulate_rbergomi_batch(
    n_paths=N_PROD,
    n_steps=N_STEPS,
    H=H,
    nu=nu,
    rho=rho,
    xi0=xi0,
    S0=S0,
    B=B,
    theta=theta,
    seed=123,
    batch_size=BATCH,
)
print(f"    Barrier survival (25k): {(S_max_full < B).mean():.4f}")
print(f"    Barrier survival (12k): {(S_max_half < B).mean():.4f}")

# ---------------------------------------------------------------------------
# Step 4 – Controlled prices: 25k monitoring
# ---------------------------------------------------------------------------
print("\n[4] Computing controlled prices…")
Y_put_full, Y_ctrl_full = compute_payoffs(S_T, S_max_full, K_grid, B)
price_25k, se_25k, ci_lo_25k, ci_hi_25k = controlled_price(
    Y_put_full, Y_ctrl_full, beta_hat, S0, B
)

# Half monitoring – same beta (fitted on full monitoring pilot)
Y_put_half, Y_ctrl_half = compute_payoffs(S_T, S_max_half, K_grid, B)
price_12k, se_12k, ci_lo_12k, ci_hi_12k = controlled_price(
    Y_put_half, Y_ctrl_half, beta_hat, S0, B
)

# ---------------------------------------------------------------------------
# Step 5 – Error metrics and monitoring sensitivity
# ---------------------------------------------------------------------------
print("\n[5] Computing error metrics and monitoring sensitivity…")

# Analytical formulas return 0 for K >= B by construction (not an approximation),
# so restrict error comparison to strikes where the formulas are valid (K < B).
valid_mask = K_grid < B
K_valid = K_grid[valid_mask]
df_errors = compute_errors(
    {"BS": p_bs[valid_mask], "RoughBergomi": p_rB[valid_mask]},
    price_25k[valid_mask],
    K_valid,
)

# Monitoring sensitivity – conservative se from summary statistics
df_monitoring = monitoring_sensitivity(price_25k, price_12k, K_grid, se_25k, se_12k)

# Actual paired SE of the difference (paths are shared → high correlation)
diff_payoffs = Y_put_full - Y_put_half   # (N_PROD, n_strikes)
actual_paired_se = diff_payoffs.std(axis=0, ddof=1) / np.sqrt(N_PROD)
max_actual_paired_se = float(actual_paired_se.max())

# ---------------------------------------------------------------------------
# Step 6 – Save CSVs
# ---------------------------------------------------------------------------
print("\n[6] Saving results CSVs…")

df_prices = pd.DataFrame(
    {
        "Strike": K_grid,
        "MC_Price": price_25k,
        "CI_Lo": ci_lo_25k,
        "CI_Hi": ci_hi_25k,
        "SE": se_25k,
        "BS_Price": p_bs,
        "RoughBergomi_Price": p_rB,
    }
)
prices_path = os.path.join(RESULTS_DIR, "exp3_prices.csv")
errors_path = os.path.join(RESULTS_DIR, "exp3_errors.csv")
monitoring_path = os.path.join(RESULTS_DIR, "exp3_monitoring.csv")

df_prices.to_csv(prices_path, index=False)
df_errors.to_csv(errors_path, index=False)
df_monitoring.to_csv(monitoring_path, index=False)
print(f"    {prices_path}")
print(f"    {errors_path}")
print(f"    {monitoring_path}")

# ---------------------------------------------------------------------------
# Step 7 – Plots
# ---------------------------------------------------------------------------
print("\n[7] Generating plots…")

# --- Plot 1: Prices vs Strike with CI band ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(
    K_grid, ci_lo_25k, ci_hi_25k,
    alpha=0.25, color="gray", label="MC 95% CI (25k steps)",
)
ax.plot(K_grid, price_25k, "o-", color="black", ms=4, lw=1.5, label="MC (25k steps)")
ax.plot(K_grid, p_bs,  "s--", color="steelblue",  ms=5, lw=1.5, label="Black-Scholes")
ax.plot(K_grid, p_rB,  "^-",  color="firebrick",  ms=5, lw=1.5, label="Rough Bergomi")
ax.set_xlabel("Strike K", fontsize=12)
ax.set_ylabel("Price", fontsize=12)
ax.set_title(
    f"Rough Bergomi Up-and-Out Put  (H={H}, ν={nu}, ρ={rho}, ξ₀={xi0}, "
    f"S₀={S0:.0f}, B={B:.0f}, T={theta})",
    fontsize=11,
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
prices_fig = os.path.join(FIGURES_DIR, "exp3_prices.png")
fig.savefig(prices_fig, dpi=150)
plt.close(fig)
print(f"    {prices_fig}")

# --- Plot 2: Signed errors vs Strike ---
fig, ax = plt.subplots(figsize=(10, 6))
method_styles = {
    "BS": ("steelblue", "s--", "Black-Scholes"),
    "RoughBergomi": ("firebrick", "^-", "Rough Bergomi"),
}
for method, (color, style, label) in method_styles.items():
    sub = df_errors[df_errors["Method"] == method].sort_values("Strike")
    ax.plot(
        sub["Strike"], sub["SignedError"],
        style, color=color, ms=5, lw=1.5, label=label,
    )
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xlabel("Strike K", fontsize=12)
ax.set_ylabel("Signed Error (Analytical − MC)", fontsize=12)
ax.set_title("Rough Bergomi: Analytical vs MC Signed Errors", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
errors_fig = os.path.join(FIGURES_DIR, "exp3_errors.png")
fig.savefig(errors_fig, dpi=150)
plt.close(fig)
print(f"    {errors_fig}")

# ---------------------------------------------------------------------------
# Step 8 – Summary metrics
# ---------------------------------------------------------------------------
bs_row = df_errors[df_errors["Method"] == "BS"].iloc[0]
rB_row = df_errors[df_errors["Method"] == "RoughBergomi"].iloc[0]
max_se = float(se_25k.max())
mon_max_diff = df_monitoring.attrs["max_abs_diff"]

print("\n" + "=" * 60)
print("Summary Metrics")
print("=" * 60)
print(f"  BS MAE:                   {bs_row['MAE']:.5f}")
print(f"  Rough-correction MAE:     {rB_row['MAE']:.5f}")
print(f"  BS max AE:                {bs_row['MaxAE']:.5f}")
print(f"  Rough-correction max AE:  {rB_row['MaxAE']:.5f}")
print(f"  Largest MC SE:            {max_se:.5f}")
print(f"  Monitoring max change:    {mon_max_diff:.5f}")
print(f"  Monitoring paired SE:     {max_actual_paired_se:.5f}  (actual paired)")

print("\nExpected reference metrics:")
print("  BS MAE              ~0.27126")
print("  Rough-correction MAE ~0.10319")
print("  BS max AE           ~0.52969")
print("  Rough-correction max AE ~0.18689")
print("  Largest MC SE       ~0.00649")
print("  Monitoring max change ~0.00005")
print("  Monitoring paired SE  ~0.00003")
print("=" * 60)
