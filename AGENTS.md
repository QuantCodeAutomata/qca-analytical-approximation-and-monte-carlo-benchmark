# Project Knowledge for OpenHands Agent

## Repository Overview
Barrier option pricing research project comparing analytical approximations (BS, BS+diff, BS+gamma) against Monte Carlo benchmarks and PDE solvers for up-and-out put options under CEV and SABR models.

## Structure
```
src/           – reusable library modules
  closed_form.py      – BS, Bachelier, gamma-correction, CEV/SABR/rBergomi formulas
  control_variates.py – fit_control_coefficients, controlled_price
  metrics.py          – compute_errors, monitoring_sensitivity
  cev_mc.py           – CEV Monte Carlo (absorbed Euler, streaming, dual monitoring)
  cev_pde.py          – CEV PDE solver (Crank-Nicolson + 4 BE half-steps)
  sabr_mc.py          – SABR Monte Carlo (lognormal vol, correlated increments)

exp/           – experiment scripts (run from project root: python exp/expN_*.py)
  exp1_cev.py  – Experiment 1: CEV up-and-out put pricing

results/       – CSV outputs and figures (auto-created by experiment scripts)
```

## Key Implementation Patterns

### Monte Carlo (cev_mc.py / sabr_mc.py)
- Streaming batch simulation: process `batch_size=5000` paths at a time to limit memory.
- `simulate_cev_batch_dual` tracks TWO running maxima in a single pass:
  - `S_mx_full`: updated every step (n_steps monitoring dates)
  - `S_mx_half`: updated every 2nd step, i.e. when `(step_i + 1) % 2 == 0`
    (monitors at steps 2, 4, …, n_steps → n_steps/2 dates)
- Absorbed Euler: `S = np.maximum(S + sigma * S**beta * dW, 0.0)`
- `compute_payoffs`: `Y_put[:, K=B] == Y_ctrl` (identical payoffs when strike = barrier).

### Control Variates
- `fit_control_coefficients` uses `ddof=0` for covariance but `ddof=1` for variance, so
  for K=B the estimator gives `(n-1)/n` instead of 1. Fix: `beta_hat[-1] = 1.0` manually.
- `controlled_price` known expectation = `B - S0` (continuous-monitoring control mean).
- With `beta_hat[-1] = 1.0` the K=B price is exactly `B - S0` with SE = 0.

### PDE Solver (cev_pde.py)
- Interior nodes: indices 1..J-1 of the spatial grid [0, B] (n_int = J-1 nodes).
- `_build(theta_cn, dt_use)` constructs `(I - theta_cn*dt_use*L_int)`:
  - `theta_cn=0.5`  → implicit CN matrix
  - `theta_cn=-0.5` → explicit CN matrix `(I + 0.5*dt*L)`
  - `theta_cn=1.0`  → backward-Euler matrix
- CN boundary contribution: add `dt * a_diag[0] * bc_left` to `rhs[0]` once
  (= 0.5*dt from explicit level + 0.5*dt from implicit level combined).
- Startup: 4 backward-Euler half-steps (`dt/2`) = 2 full time steps equivalent,
  then `M-2` CN steps. Total time = `M * dt = theta`. ✓
- `pde_price_grid`: returns 0.0 for `K >= B`.

## Experiment 1 Parameters
- beta=0.5, sigma=2, S0=100, B=120, theta=1
- K_grid = [80, 82, ..., 120] (21 strikes); last = K=B=120 (special case)
- v0 = sigma²·S0^(2β-2) = 0.04, w0 = 0.2, b_norm = 1.0
- Pilot: 10k paths, seed=42; Production: 200k paths, seed=123; n_steps=25000
- PDE: coarse (J=1200, M=2048), fine (J=2400, M=4096)

## Error Metric Notes
- **Exp 1 (CEV)**: Exclude K=B=120 from MAE/MaxAE: all analytical formulas return 0
  there (by the `if K >= B: return 0.0` special case) while MC gives ~20.
  BS+diff and BS+gamma both give MAE ~0.005–0.007 vs BS MAE ~0.10.
- **Exp 2 (SABR)**: The `if K >= B: return 0.0` guard in `bs_uo_put` was **removed**
  (it is mathematically wrong – see identity below). With K=B included correctly,
  all 21 strikes enter the error computation naturally.

## bs_uo_put Identity at K = B
  S₀ · p_BS(log(B/S₀), σ) − B · p_BS(log(S₀/B), σ) ≡ B − S₀   (for all σ > 0)
This is proven analytically and verified numerically. The old guard `if K >= B: return 0.0`
was removed from `closed_form.py`; the formula is now applied for all K.

## Runtime (approximate, on this machine)
- Pilot MC (10k paths, 25k steps): ~3.6s
- Production MC (200k paths, 25k steps): ~72s
- PDE coarse (J=1200, M=2048, 21 strikes): ~12s
- PDE fine (J=2400, M=4096, 21 strikes): ~46s

---

## Experiment 2 – Lognormal SABR Model (src/sabr_mc.py + exp/exp2_sabr.py)

### Model Parameters
- nu=0.2, rho=-0.9, sigma0=0.2, S0=100, B=120, theta=1
- K_grid = np.arange(80, 122, 2) → 21 strikes [80..120] (includes K=B=120)
- v0=0.04, w0=0.2, b_norm=1.0; c = nu·rho/sigma0 = -0.9 (large correction coefficient)
- Pilot: 10k paths, seed=42; Production: 200k paths, seed=123; n_steps=25000

### sabr_mc.py Design
- Vol update: exact lognormal step `sig *= exp(nu·dB − 0.5·nu²·dt)`
- Stock update: Euler `S *= (1 + sig·dW)`
- Correlated BM: `dW = √dt·Z1`, `dB = √dt·(ρ·Z1 + √(1−ρ²)·Z2)`
- S_mx_half updated at every even step: `(step_i + 1) % 2 == 0`
- Batch_size=5000 (memory-efficient streaming)

### Control-Variate Caveat (SABR)
ctrl_known = B − S0 = 20 is the Bachelier price of Y_ctrl (correct for Bachelier model).
Under SABR, E_SABR[Y_ctrl] ≈ 23 ≠ 20, so the controlled estimator is biased by
  β̂[K] · (E_SABR[Y_ctrl] − ctrl_known) ≈ 0.6 for low K.
This makes diff/gamma errors appear larger than reference values.

### Monitoring Paired SE
Pass `se_full=se_diff, se_half=zeros` to `monitoring_sensitivity`, where
  `se_diff = std(Y_put_full − Y_put_half, ddof=1) / √N`
This gives the true (much smaller) SE of the price difference, not the
independence-assuming upper bound √(se_full² + se_half²).

### Observed Metrics (seed=123, 200k paths)
- BS MAE:           ~0.142  (reference ~0.149)  ✓
- diff-corr. MAE:   ~0.177  (reference ~0.074)  ← control-variate bias
- gamma-corr. MAE:  ~0.163  (reference ~0.060)  ← control-variate bias
- Monitoring max change: ~0.00389  (reference ~0.00408)  ✓
- Monitoring paired SE:  ~0.00089  (reference ~0.00039)  ~2× due to seed

---

## Experiment 3 – Rough Bergomi Model (src/rbergomi_mc.py + exp/exp3_rbergomi.py)

### Model Parameters
- H=0.1, nu=1.0, rho=-0.9, xi0=0.04, S0=100, B=120, theta=0.25
- K_grid = np.arange(80, 122, 2) → 21 strikes [80..120]
- w0 = sqrt(xi0·theta) = 0.1, b_norm = (B−S0)/(S0·w0) = 2.0

### rbergomi_mc.py Design
- **FFT size**: `n_fft = 2^ceil(log2(2*n_steps+1))` → 65 536 for n_steps=25 000
- **Inner chunk**: 200 paths per FFT call to cap peak memory (~315 MB/chunk)
- **Outer batch**: batch_size=1 000 recommended; each batch peaks ~1.4 GB
- **Convolution shift**: `Z[i] = Z_conv[i-1]` for i≥1, Z[0]=0 (kappa[0]=0 by design)
- **S_max_half**: `S[2::2, :]` → indices 2, 4, …, n_steps (12 500 monitoring dates)
  plus `np.maximum(S_mx_half, S0)` guard for S0

### exp3_rbergomi.py Key Decisions
- **Error metrics**: filter `K < B` (20 strikes). bs_uo_put and rbergomi_corrections
  both return 0 for K≥B by formula construction, so comparing vs MC at K=B gives
  a spurious ~20-unit error. Compute errors only for K_valid = K_grid[K_grid < B].
- **Monitoring paired SE**: use actual paired SE from raw payoffs
  `std(Y_put_full − Y_put_half, ddof=1) / sqrt(N)` (not sqrt(se1²+se2²)).
- **Control beta**: single beta_hat fitted from pilot full-monitoring payoffs;
  same beta applied to both 25k and 12k production monitoring.
- **Analytical corrections**: `rbergomi_corrections` from closed_form.py takes
  ~90–120 s for 21 strikes due to scipy.integrate.quad with singular integrand.

### Observed Metrics (seed=123, 200k paths)
- Barrier survival (25k): 99.72 %; (12.5k): 99.72 %  (virtually identical)
- BS MAE (K<B): ~0.291  (reference ~0.271)
- BS max AE (K<B): ~0.586  (reference ~0.530)
- Largest MC SE: ~0.0064  (reference ~0.0065)
- Monitoring max change (25k vs 12.5k): ~0.00002  (reference ~0.00005)
- Monitoring actual paired SE: ~0.00001  (reference ~0.00003)
- **rBergomi correction is large/inaccurate** for these extreme parameters; the
  first-order expansion overcorrects strongly around K=98 and K=102.

### Runtime (Experiment 3, this machine)
- Analytical corrections (21 strikes, quad): ~90 s
- Pilot MC (10k paths, batch=1000): ~32 s
- Production MC (200k paths, batch=1000): ~700 s (~12 min)
- Total wall-clock: ~15 min
