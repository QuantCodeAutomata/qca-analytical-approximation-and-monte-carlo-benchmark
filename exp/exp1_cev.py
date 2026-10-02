"""
Experiment 1: CEV model up-and-out put pricing.

Model: dS_t = sigma * S_t^beta * dW_t  (zero drift)
Parameters: beta=0.5, sigma=2, S0=100, B=120, theta=1
Strike grid: K in {80, 82, ..., 120}  (21 strikes)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from project root or from exp/
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.closed_form import cev_corrections
from src.control_variates import fit_control_coefficients, controlled_price
from src.metrics import compute_errors, monitoring_sensitivity
from src.cev_mc import simulate_cev_batch_dual, compute_payoffs
from src.cev_pde import pde_price_grid

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BETA   = 0.5
SIGMA  = 2.0
S0     = 100.0
B      = 120.0
THETA  = 1.0

K_GRID  = np.arange(80, 122, 2, dtype=float)   # [80, 82, ..., 120], 21 strikes
N_PILOT = 10_000
N_PROD  = 200_000
N_STEPS = 25_000

# Derived quantities (informational)
V0     = SIGMA ** 2 * S0 ** (2.0 * BETA - 2.0)   # = 4 * 100^(-1) = 0.04
W0     = np.sqrt(V0 * THETA)                       # = 0.2
B_NORM = (B - S0) / (S0 * W0)                     # = 1.0

# Output directories
RESULTS_DIR  = os.path.join(_root, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Analytical approximations
# ---------------------------------------------------------------------------
print("=" * 60)
print("Experiment 1: CEV up-and-out put pricing")
print("=" * 60)
print(f"  beta={BETA}, sigma={SIGMA}, S0={S0}, B={B}, theta={THETA}")
print(f"  v0={V0:.4f}, w0={W0:.4f}, b_norm={B_NORM:.4f}")
print()

print("Step 1: Computing analytical approximations …")
cf = cev_corrections(S0, K_GRID, B, BETA, SIGMA, THETA)
p_bs    = cf["p_bs"]
p_diff  = cf["p_diff"]
p_gamma = cf["p_gamma"]
print("  Done.\n")

# ---------------------------------------------------------------------------
# 2. Pilot Monte Carlo  (seed=42, 10 k paths)
# ---------------------------------------------------------------------------
print(f"Step 2: Pilot MC  (n={N_PILOT:,}, n_steps={N_STEPS:,}, seed=42) …")
t0 = time.perf_counter()
S_T_p, S_mx_full_p, S_mx_half_p = simulate_cev_batch_dual(
    N_PILOT, N_STEPS, SIGMA, BETA, S0, B, THETA, seed=42
)
print(f"  Done in {time.perf_counter()-t0:.1f}s.\n")

# Pilot payoffs using full monitoring for coefficient fitting
Y_put_p, Y_ctrl_p = compute_payoffs(S_T_p, S_mx_full_p, K_GRID, B)

# ---------------------------------------------------------------------------
# 3. Fit control coefficients
# ---------------------------------------------------------------------------
print("Step 3: Fitting control-variate coefficients …")
beta_hat = fit_control_coefficients(Y_put_p, Y_ctrl_p)
# K=B (last strike): Y_put = Y_ctrl exactly, so beta = 1 by theory.
# The finite-pilot estimator gives (n-1)/n due to ddof mismatch; correct it.
beta_hat[-1] = 1.0
print(f"  beta_hat range: [{beta_hat.min():.4f}, {beta_hat.max():.4f}]")
print()

# ---------------------------------------------------------------------------
# 4. Production Monte Carlo  (seed=123, 200 k paths)
# ---------------------------------------------------------------------------
print(f"Step 4: Production MC  (n={N_PROD:,}, n_steps={N_STEPS:,}, seed=123) …")
t0 = time.perf_counter()
S_T, S_mx_full, S_mx_half = simulate_cev_batch_dual(
    N_PROD, N_STEPS, SIGMA, BETA, S0, B, THETA, seed=123
)
print(f"  Done in {time.perf_counter()-t0:.1f}s.\n")

# ---------------------------------------------------------------------------
# 5. Compute payoffs for both monitoring schemes
# ---------------------------------------------------------------------------
print("Step 5: Computing payoffs for 25k and 12.5k monitoring dates …")
Y_put_full, Y_ctrl_full = compute_payoffs(S_T, S_mx_full, K_GRID, B)
Y_put_half, Y_ctrl_half = compute_payoffs(S_T, S_mx_half, K_GRID, B)
print("  Done.\n")

# ---------------------------------------------------------------------------
# 6. Controlled prices for both monitoring schemes
# ---------------------------------------------------------------------------
print("Step 6: Computing controlled MC prices …")
price_full, se_full, ci_lo_full, ci_hi_full = controlled_price(
    Y_put_full, Y_ctrl_full, beta_hat, S0, B
)
price_half, se_half, ci_lo_half, ci_hi_half = controlled_price(
    Y_put_half, Y_ctrl_half, beta_hat, S0, B
)

# K=B=120 (last strike): by construction price = B - S0 = 20, SE = 0
assert abs(price_full[-1] - (B - S0)) < 1e-10, \
    f"K=B price_full={price_full[-1]:.6f}, expected {B-S0:.1f}"
assert se_full[-1] < 1e-12, f"K=B SE_full={se_full[-1]}"
print(f"  K=B check: price_full[last]={price_full[-1]:.4f}, SE={se_full[-1]:.2e}  ✓")
print()

# ---------------------------------------------------------------------------
# 7. Error metrics
# ---------------------------------------------------------------------------
print("Step 7: Computing error metrics …")

# Exclude K=B from summary statistics (it is a special case where all
# analytical formulas return 0 but the true MC price is B-S0=20)
K_inner   = K_GRID[:-1]
mc_inner  = price_full[:-1]
analytical_inner = {
    "BS":       p_bs[:-1],
    "BS+diff":  p_diff[:-1],
    "BS+gamma": p_gamma[:-1],
}
errors_df = compute_errors(analytical_inner, mc_inner, K_inner)

# Append the K=B row manually so it appears in the saved CSV
rows_kB = []
for method, arr in [("BS", p_bs), ("BS+diff", p_diff), ("BS+gamma", p_gamma)]:
    se_kB = price_full[-1] - arr[-1]
    rows_kB.append({
        "Strike": K_GRID[-1],
        "Method": method,
        "Price": arr[-1],
        "MC_Price": price_full[-1],
        "SignedError": se_kB,
        "AbsError": abs(se_kB),
        "MAE": float("nan"),
        "MaxAE": float("nan"),
    })
errors_df = pd.concat([errors_df, pd.DataFrame(rows_kB)], ignore_index=True)

monitor_df = monitoring_sensitivity(
    price_full, price_half, K_GRID, se_full, se_half
)
print("  Done.\n")

# ---------------------------------------------------------------------------
# 8. PDE prices  (two grids per strike)
# ---------------------------------------------------------------------------
print("Step 8: Running PDE solver …")

print("  Coarse grid (J=1200, M=2048) …")
t0 = time.perf_counter()
pde_coarse = pde_price_grid(K_GRID, S0, B, SIGMA, BETA, THETA, J=1200, M=2048)
print(f"  Done in {time.perf_counter()-t0:.1f}s.")

print("  Fine grid (J=2400, M=4096) …")
t0 = time.perf_counter()
pde_fine = pde_price_grid(K_GRID, S0, B, SIGMA, BETA, THETA, J=2400, M=4096)
print(f"  Done in {time.perf_counter()-t0:.1f}s.\n")

# ---------------------------------------------------------------------------
# 9. Summary tables
# ---------------------------------------------------------------------------
print("=" * 60)
print("PRICES SUMMARY  (full 25k monitoring, controlled MC)")
print("=" * 60)
prices_df = pd.DataFrame({
    "Strike":     K_GRID,
    "MC_price":   np.round(price_full, 4),
    "MC_SE":      np.round(se_full, 4),
    "CI_lo":      np.round(ci_lo_full, 4),
    "CI_hi":      np.round(ci_hi_full, 4),
    "BS":         np.round(p_bs, 4),
    "BS+diff":    np.round(p_diff, 4),
    "BS+gamma":   np.round(p_gamma, 4),
    "PDE_coarse": np.round(pde_coarse, 4),
    "PDE_fine":   np.round(pde_fine, 4),
})
print(prices_df.to_string(index=False))
print()

print("=" * 60)
print("ERROR METRICS  (analytic − MC, excluding K=B=120)")
print("=" * 60)
summary_methods = errors_df[errors_df["Strike"] < B].groupby("Method").agg(
    MAE=("AbsError", "mean"), MaxAE=("AbsError", "max")
).reset_index()
print(summary_methods.to_string(index=False))
print()

print("=" * 60)
print("MONITORING SENSITIVITY  (25k vs 12.5k dates)")
print("=" * 60)
print(monitor_df[["Strike", "Price_25k", "Price_12k", "Diff", "AbsDiff"]].to_string(index=False))
print(f"\n  Max |price_25k - price_12k| = {monitor_df.attrs['max_abs_diff']:.6f}")
print()

print("=" * 60)
print("PDE CONVERGENCE  (vs fine-grid PDE and MC, excluding K=B=120)")
print("=" * 60)
pde_df = pd.DataFrame({
    "Strike":     K_GRID,
    "PDE_coarse": np.round(pde_coarse, 5),
    "PDE_fine":   np.round(pde_fine, 5),
    "MC_price":   np.round(price_full, 5),
    "err_coarse": np.round(pde_coarse - pde_fine, 6),
    "err_fine_vs_MC": np.round(pde_fine - price_full, 6),
})
print(pde_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
print("Step 10: Creating plots …")

_cm = plt.cm.tab10.colors

# --- (a) Prices vs strike ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(K_GRID, ci_lo_full, ci_hi_full, alpha=0.25, color=_cm[0],
                label="MC 95% CI (25k dates)")
ax.plot(K_GRID, price_full,  color=_cm[0], lw=2,    label="MC controlled (25k)")
ax.plot(K_GRID, price_half,  color=_cm[0], lw=1.5,  ls="--", label="MC controlled (12.5k)")
ax.plot(K_GRID, p_bs,        color=_cm[1], lw=1.5,  label="BS")
ax.plot(K_GRID, p_diff,      color=_cm[2], lw=1.5,  label="BS+diff")
ax.plot(K_GRID, p_gamma,     color=_cm[3], lw=1.5,  label="BS+gamma")
ax.plot(K_GRID, pde_fine,    color=_cm[4], lw=1.5,  ls="-.", label="PDE fine")
ax.set_xlabel("Strike K")
ax.set_ylabel("Price")
ax.set_title(
    f"CEV up-and-out put prices\n"
    r"$\beta$" + f"={BETA}, σ={SIGMA}, S₀={S0}, B={B}, θ={THETA}"
)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(K_GRID[0], K_GRID[-1])
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp1_prices.png"), dpi=150)
plt.close(fig)
print("  Saved exp1_prices.png")

# --- (b) Signed errors vs strike (excluding K=B) ---
K_plot   = K_GRID[:-1]
err_bs   = p_bs[:-1]    - price_full[:-1]
err_diff = p_diff[:-1]  - price_full[:-1]
err_gam  = p_gamma[:-1] - price_full[:-1]
err_pde  = pde_fine[:-1] - price_full[:-1]

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.plot(K_plot, err_bs,   color=_cm[1], lw=1.5, label="BS − MC")
ax.plot(K_plot, err_diff, color=_cm[2], lw=1.5, label="BS+diff − MC")
ax.plot(K_plot, err_gam,  color=_cm[3], lw=1.5, label="BS+gamma − MC")
ax.plot(K_plot, err_pde,  color=_cm[4], lw=1.5, ls="-.", label="PDE fine − MC")
ax.fill_between(K_plot, -1.96 * se_full[:-1], 1.96 * se_full[:-1],
                alpha=0.15, color=_cm[0], label="±1.96×SE (MC)")
ax.set_xlabel("Strike K")
ax.set_ylabel("Signed error (approx − MC)")
ax.set_title(
    "CEV up-and-out put: signed pricing errors\n"
    r"$\beta$" + f"={BETA}, σ={SIGMA}, S₀={S0}, B={B}, θ={THETA}"
)
ax.legend(fontsize=8)
ax.set_xlim(K_plot[0], K_plot[-1])
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp1_errors.png"), dpi=150)
plt.close(fig)
print("  Saved exp1_errors.png\n")

# ---------------------------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------------------------
print("Step 11: Saving CSV results …")

prices_df.to_csv(os.path.join(RESULTS_DIR, "exp1_prices.csv"), index=False)
print("  Saved exp1_prices.csv")

errors_df.to_csv(os.path.join(RESULTS_DIR, "exp1_errors.csv"), index=False)
print("  Saved exp1_errors.csv")

monitor_df.to_csv(os.path.join(RESULTS_DIR, "exp1_monitoring.csv"), index=False)
print("  Saved exp1_monitoring.csv")

pde_df.to_csv(os.path.join(RESULTS_DIR, "exp1_pde.csv"), index=False)
print("  Saved exp1_pde.csv")

print("\nExperiment 1 complete.")
